import argparse
import ast
import hashlib
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MANUAL_DIR = BASE_DIR / "手册"
IMAGE_DIR = MANUAL_DIR / "插图"
RAG_CACHE_DIR = MANUAL_DIR / "rag_cache"
MANUAL_ALIAS_PATH = BASE_DIR / "manual_aliases.yaml"

DEFAULT_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_EMBEDDING_MODEL_DIR = BASE_DIR / "Qwen3-VL-Embedding-2B"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 30
DEFAULT_BATCH_SIZE = 4
DEFAULT_QUERY_PROMPT = "Retrieve relevant product manual passages for the user's question."
ENGLISH_SUMMARY_PRODUCT = "英文汇总"
MANUAL_SUFFIX = "手册"
PIC_TOKEN = "<PIC>"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CACHE_VERSION = 2


@dataclass
class ManualEntry:
    content: str
    image_names: list[str]


@dataclass
class ChunkMetaData:
    entry_index: int
    start: int
    end: int
    char_count: int
    image_count: int
    chunking_strategy: str
    heading: str = ""


@dataclass
class ManualChunk:
    text: str
    entry_index: int
    start: int
    end: int
    image_names: list[str] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)
    metadata: ChunkMetaData | None = None


@dataclass
class RetrievalResult:
    chunk: ManualChunk
    score: float


_default_rag: "ManualRAG | None" = None


def get_default_rag() -> "ManualRAG":
    """Return one reusable RAG instance so the embedding model is loaded once."""
    global _default_rag
    if _default_rag is None:
        _default_rag = ManualRAG()
    return _default_rag


def download_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    model_dir: Path = DEFAULT_EMBEDDING_MODEL_DIR,
) -> Path:
    """Download Qwen3-VL-Embedding into this project directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少 huggingface_hub，请先安装：pip install huggingface-hub") from exc

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    return model_dir


class ManualRAG:
    """Vector-only RAG over one known manual using Qwen3-VL-Embedding."""

    def __init__(
        self,
        embedding_model_dir: Path = DEFAULT_EMBEDDING_MODEL_DIR,
        image_dir: Path = IMAGE_DIR,
        cache_dir: Path = RAG_CACHE_DIR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_images_per_chunk: int = 0,
        query_prompt: str = DEFAULT_QUERY_PROMPT,
        device: str = "cpu",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size。")
        if max_images_per_chunk < 0:
            raise ValueError("max_images_per_chunk 不能小于 0。")

        self.embedding_model_dir = embedding_model_dir
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.max_images_per_chunk = max_images_per_chunk
        self.query_prompt = query_prompt
        self.device = device
        self._embedder: Any | None = None

    def retrieve(
        self,
        question: str,
        manual_path: Path,
        product_name: str | None = None,
        top_k: int = 5,
        neighbor_count: int = 0,
        rebuild_cache: bool = False,
    ) -> list[RetrievalResult]:
        index = self.load_or_build_index(
            manual_path=manual_path,
            product_name=product_name,
            rebuild_cache=rebuild_cache,
        )
        chunks: list[ManualChunk] = index["chunks"]
        document_chunk_indexes: list[int] = index["document_chunk_indexes"]
        document_vectors: list[list[float]] = index["document_vectors"]
        if not chunks or not document_vectors:
            return []

        question_vector = self._encode_query(question)

        best_by_chunk: dict[int, RetrievalResult] = {}
        for chunk_index, document_vector in zip(document_chunk_indexes, document_vectors):
            chunk = chunks[chunk_index]
            score = cosine_similarity(question_vector, document_vector)
            current = best_by_chunk.get(chunk_index)
            if current is None or score > current.score:
                best_by_chunk[chunk_index] = RetrievalResult(chunk=chunk, score=score)

        top_results = sorted(best_by_chunk.values(), key=lambda item: item.score, reverse=True)[:top_k]
        if neighbor_count <= 0:
            return top_results
        return expand_with_neighbor_chunks(top_results, chunks, neighbor_count)

    def load_or_build_index(
        self,
        manual_path: Path,
        product_name: str | None = None,
        rebuild_cache: bool = False,
    ) -> dict[str, Any]:
        entries = load_manual_entries(manual_path)
        entry_indexes = selected_entry_indexes(manual_path, product_name, len(entries))
        chunks = split_manual_entries(
            entries,
            entry_indexes=entry_indexes,
            image_dir=self.image_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        metadata = self._cache_metadata(manual_path, product_name, entry_indexes, chunks)
        cache_path = self._cache_path(manual_path, product_name)

        if not rebuild_cache:
            cached = self._load_cache(cache_path, metadata)
            if cached is not None:
                return cached

        document_chunk_indexes: list[int] = []
        documents: list[str | dict[str, str]] = []
        for chunk_index, chunk in enumerate(chunks):
            for document in self._documents_from_chunk(chunk):
                document_chunk_indexes.append(chunk_index)
                documents.append(document)

        document_vectors = self._encode_documents(documents) if documents else []
        payload = {
            "metadata": metadata,
            "chunks": chunks,
            "document_chunk_indexes": document_chunk_indexes,
            "document_vectors": document_vectors,
        }
        self._save_cache(cache_path, payload)
        return payload

    def build_chunks(self, manual_path: Path, product_name: str | None = None) -> list[ManualChunk]:
        """Expose chunking for quick checks without loading the embedding model."""
        entries = load_manual_entries(manual_path)
        return split_manual_entries(
            entries,
            entry_indexes=selected_entry_indexes(manual_path, product_name, len(entries)),
            image_dir=self.image_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def warmup_all(
        self,
        manual_dir: Path = MANUAL_DIR,
        rebuild_cache: bool = False,
    ) -> list[dict[str, Any]]:
        """Build vector caches for every manual.

        English summary is cached per sub-product, because routing already knows
        product_name before RAG retrieval.
        """
        warmed = []
        for manual_path in manual_files(manual_dir):
            manual_product_name = product_name_from_manual(manual_path)
            product_names: list[str | None]
            if manual_product_name == ENGLISH_SUMMARY_PRODUCT:
                product_names = english_summary_product_names()
            else:
                product_names = [None]

            for product_name in product_names:
                index = self.load_or_build_index(
                    manual_path=manual_path,
                    product_name=product_name,
                    rebuild_cache=rebuild_cache,
                )
                warmed.append(
                    {
                        "manual_path": manual_path,
                        "product_name": product_name,
                        "chunk_count": len(index["chunks"]),
                        "document_count": len(index["document_vectors"]),
                    }
                )

        return warmed

    def _documents_from_chunk(self, chunk: ManualChunk) -> list[str | dict[str, str]]:
        image_paths = chunk.image_paths[: self.max_images_per_chunk]
        if image_paths:
            return [
                {
                    "text": chunk.text,
                    "image": str(image_path),
                }
                for image_path in image_paths
            ]
        return [chunk.text]

    def _encode_query(self, question: str) -> list[float]:
        vectors = self._encode([question], prompt=self.query_prompt)
        return vectors[0]

    def _encode_documents(self, documents: list[str | dict[str, str]]) -> list[list[float]]:
        return self._encode(documents)

    def _encode(
        self,
        inputs: list[str | dict[str, str]],
        prompt: str | None = None,
    ) -> list[list[float]]:
        if self._embedder is None:
            self._embedder = self._load_embedder()

        encode_kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": False,
            "show_progress_bar": False,
        }
        if prompt:
            encode_kwargs["prompt"] = prompt

        vectors = self._embedder.encode(inputs, **encode_kwargs)
        return [list(map(float, vector)) for vector in vectors]

    def _load_embedder(self) -> Any:
        if not self.embedding_model_dir.exists():
            raise FileNotFoundError(
                f"未找到本地 Qwen3-VL-Embedding 模型：{self.embedding_model_dir}\n"
                "可执行：python rag.py --download-model"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 sentence-transformers，请先安装：pip install sentence-transformers"
            ) from exc

        return SentenceTransformer(str(self.embedding_model_dir), device=self.device)

    def _cache_path(self, manual_path: Path, product_name: str | None) -> Path:
        product_part = product_name or "all"
        key = "|".join(
            [
                str(manual_path.resolve()),
                product_part,
                str(self.chunk_size),
                str(self.chunk_overlap),
                str(self.max_images_per_chunk),
                str(self.embedding_model_dir.resolve()),
            ]
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        safe_product = safe_cache_name(product_part)
        return self.cache_dir / f"{manual_path.stem}.{safe_product}.{digest}.pkl"

    def _cache_metadata(
        self,
        manual_path: Path,
        product_name: str | None,
        entry_indexes: list[int],
        chunks: list[ManualChunk],
    ) -> dict[str, Any]:
        stat = manual_path.stat()
        return {
            "cache_version": CACHE_VERSION,
            "manual_path": str(manual_path.resolve()),
            "manual_mtime_ns": stat.st_mtime_ns,
            "manual_size": stat.st_size,
            "product_name": product_name,
            "entry_indexes": entry_indexes,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_images_per_chunk": self.max_images_per_chunk,
            "embedding_model_dir": str(self.embedding_model_dir.resolve()),
            "image_fingerprints": image_fingerprints(chunks),
        }

    def _load_cache(self, cache_path: Path, metadata: dict[str, Any]) -> dict[str, Any] | None:
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("rb") as file:
                payload = pickle.load(file)
        except (OSError, pickle.PickleError, EOFError, AttributeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("metadata") != metadata:
            return None
        payload["chunks"] = [
            deserialize_chunk(chunk)
            for chunk in payload.get("chunks", [])
        ]
        return payload

    def _save_cache(self, cache_path: Path, payload: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        serializable_payload = {
            **payload,
            "chunks": [
                serialize_chunk(chunk)
                for chunk in payload.get("chunks", [])
            ],
        }
        with cache_path.open("wb") as file:
            pickle.dump(serializable_payload, file)


def load_manual_entries(manual_path: Path) -> list[ManualEntry]:
    try:
        raw_text = manual_path.read_text(encoding="utf-8")
    except OSError:
        return []

    parsed_items: list[Any] = []
    try:
        parsed_items.append(ast.literal_eval(raw_text))
    except (SyntaxError, ValueError):
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed_items.append(ast.literal_eval(line))
            except (SyntaxError, ValueError):
                continue

    entries = []
    for item in parsed_items:
        if not isinstance(item, list) or len(item) < 2:
            continue

        image_names = []
        if isinstance(item[1], list):
            image_names = [
                str(image_name)
                for image_name in item[1]
                if str(image_name).strip()
            ]

        entries.append(
            ManualEntry(
                content=str(item[0]),
                image_names=image_names,
            )
        )

    return entries


def selected_entry_indexes(
    manual_path: Path,
    product_name: str | None,
    entry_count: int,
) -> list[int]:
    if entry_count <= 0:
        return []

    manual_product_name = product_name_from_manual(manual_path)
    if manual_product_name != ENGLISH_SUMMARY_PRODUCT or not product_name:
        return list(range(entry_count))

    entry_index = english_summary_entry_index(product_name)
    if entry_index is None or entry_index >= entry_count:
        return list(range(entry_count))
    return [entry_index]


def english_summary_entry_index(product_name: str) -> int | None:
    product_names = english_summary_product_names()
    try:
        return product_names.index(product_name)
    except ValueError:
        return None


def english_summary_product_names() -> list[str]:
    try:
        import yaml
    except ImportError:
        return []

    if not MANUAL_ALIAS_PATH.exists():
        return []

    with MANUAL_ALIAS_PATH.open("r", encoding="utf-8") as file:
        aliases = yaml.safe_load(file) or {}

    english_aliases = aliases.get(ENGLISH_SUMMARY_PRODUCT)
    if not isinstance(english_aliases, dict):
        return []

    return [str(name) for name in english_aliases.keys()]


def split_manual_entries(
    entries: list[ManualEntry],
    entry_indexes: list[int] | None = None,
    image_dir: Path = IMAGE_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ManualChunk]:
    chunks: list[ManualChunk] = []
    selected_indexes = entry_indexes if entry_indexes is not None else list(range(len(entries)))

    for entry_index in selected_indexes:
        if entry_index >= len(entries):
            continue
        entry = entries[entry_index]
        content = entry.content
        if not content:
            continue

        pic_positions = [
            match.start()
            for match in re.finditer(re.escape(PIC_TOKEN), content)
        ]

        for start, end, strategy in paragraph_chunk_spans(content, chunk_size, chunk_overlap):
            chunks.append(
                build_manual_chunk(
                    content=content,
                    entry_index=entry_index,
                    start=start,
                    end=end,
                    strategy=strategy,
                    entry_image_names=entry.image_names,
                    pic_positions=pic_positions,
                    image_dir=image_dir,
                )
            )

    return chunks


def build_manual_chunk(
    content: str,
    entry_index: int,
    start: int,
    end: int,
    strategy: str,
    entry_image_names: list[str],
    pic_positions: list[int],
    image_dir: Path,
) -> ManualChunk:
    image_names = image_names_for_span(
        image_names=entry_image_names,
        pic_positions=pic_positions,
        start=start,
        end=end,
    )
    image_paths = [
        image_path
        for image_name in image_names
        if (image_path := resolve_image_path(image_name, image_dir)) is not None
    ]
    metadata = ChunkMetaData(
        entry_index=entry_index,
        start=start,
        end=end,
        char_count=end - start,
        image_count=len(image_names),
        chunking_strategy=strategy,
        heading=extract_chunk_heading(content[start:end]),
    )
    return ManualChunk(
        text=content[start:end],
        entry_index=entry_index,
        start=start,
        end=end,
        image_names=image_names,
        image_paths=image_paths,
        metadata=metadata,
    )


def paragraph_chunk_spans(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush_current() -> None:
        nonlocal current_start, current_end
        if current_start is not None and current_end is not None:
            spans.append((current_start, current_end, "paragraph"))
        current_start = None
        current_end = None

    for line_start, line_end in non_empty_line_spans(content):
        if line_end - line_start > chunk_size:
            flush_current()
            spans.extend(
                (start, end, "paragraph_window")
                for start, end in split_long_span(content, line_start, line_end, chunk_size, chunk_overlap)
            )
            continue

        if current_start is None:
            current_start = line_start
            current_end = line_end
            continue

        if line_end - current_start <= chunk_size:
            current_end = line_end
            continue

        flush_current()
        current_start = line_start
        current_end = line_end

    flush_current()
    return spans


def non_empty_line_spans(content: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\n\r]+(?:\r?\n)?", content):
        line = match.group(0)
        if line.strip():
            spans.append((match.start(), match.end()))
    return spans


def split_long_span(
    content: str,
    start: int,
    end: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + chunk_size, end)
        split_at = limit
        if limit < end:
            relative_text = content[cursor:limit]
            break_positions = [
                relative_text.rfind(separator)
                for separator in ("。", "！", "？", "；", ";", ".", "!", "?", "\n")
            ]
            best_break = max(break_positions)
            if best_break >= chunk_size // 3:
                split_at = cursor + best_break + 1
        spans.append((cursor, split_at))
        if split_at >= end:
            break
        cursor = max(split_at - chunk_overlap, cursor + 1)
    return spans


def extract_chunk_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def image_names_for_span(
    image_names: list[str],
    pic_positions: list[int],
    start: int,
    end: int,
) -> list[str]:
    selected = []
    for pic_index, pic_position in enumerate(pic_positions):
        if start <= pic_position < end and pic_index < len(image_names):
            selected.append(image_names[pic_index])
    return selected


def resolve_image_path(image_name: str, image_dir: Path = IMAGE_DIR) -> Path | None:
    candidate = Path(image_name)
    candidates = []
    if candidate.suffix:
        candidates.append(candidate)
    else:
        candidates.extend(candidate.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)

    for item in candidates:
        full_path = item if item.is_absolute() else image_dir / item
        if full_path.exists():
            return full_path
    return None


def image_fingerprints(chunks: list[ManualChunk]) -> list[tuple[str, int, int]]:
    paths = sorted({path.resolve() for chunk in chunks for path in chunk.image_paths})
    fingerprints = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        fingerprints.append((str(path), stat.st_mtime_ns, stat.st_size))
    return fingerprints


def expand_with_neighbor_chunks(
    top_results: list[RetrievalResult],
    chunks: list[ManualChunk],
    neighbor_count: int,
) -> list[RetrievalResult]:
    chunk_positions = {
        chunk_key(chunk): index
        for index, chunk in enumerate(chunks)
    }
    top_scores = {
        chunk_key(result.chunk): result.score
        for result in top_results
    }

    selected_indexes: set[int] = set()
    for result in top_results:
        center_index = chunk_positions.get(chunk_key(result.chunk))
        if center_index is None:
            continue
        entry_index = result.chunk.entry_index
        for offset in range(-neighbor_count, neighbor_count + 1):
            candidate_index = center_index + offset
            if candidate_index < 0 or candidate_index >= len(chunks):
                continue
            if chunks[candidate_index].entry_index != entry_index:
                continue
            selected_indexes.add(candidate_index)

    expanded_results = []
    for index in sorted(
        selected_indexes,
        key=lambda item: (chunks[item].entry_index, chunks[item].start, chunks[item].end),
    ):
        chunk = chunks[index]
        expanded_results.append(
            RetrievalResult(
                chunk=chunk,
                score=top_scores.get(chunk_key(chunk), 0.0),
            )
        )
    return expanded_results


def chunk_key(chunk: ManualChunk) -> tuple[int, int, int]:
    return (chunk.entry_index, chunk.start, chunk.end)


def serialize_chunk(chunk: ManualChunk) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "entry_index": chunk.entry_index,
        "start": chunk.start,
        "end": chunk.end,
        "image_names": chunk.image_names,
        "image_paths": [str(path) for path in chunk.image_paths],
        "metadata": serialize_chunk_metadata(chunk.metadata),
    }


def deserialize_chunk(chunk: Any) -> ManualChunk:
    if isinstance(chunk, ManualChunk):
        return chunk
    if not isinstance(chunk, dict):
        return ManualChunk(text="", entry_index=0, start=0, end=0)
    metadata = chunk.get("metadata")
    return ManualChunk(
        text=str(chunk.get("text", "")),
        entry_index=int(chunk.get("entry_index", 0)),
        start=int(chunk.get("start", 0)),
        end=int(chunk.get("end", 0)),
        image_names=[
            str(image_name)
            for image_name in chunk.get("image_names", [])
        ],
        image_paths=[
            Path(image_path)
            for image_path in chunk.get("image_paths", [])
        ],
        metadata=deserialize_chunk_metadata(metadata, chunk),
    )


def serialize_chunk_metadata(metadata: ChunkMetaData | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {
        "entry_index": metadata.entry_index,
        "start": metadata.start,
        "end": metadata.end,
        "char_count": metadata.char_count,
        "image_count": metadata.image_count,
        "chunking_strategy": metadata.chunking_strategy,
        "heading": metadata.heading,
    }


def deserialize_chunk_metadata(
    metadata: Any,
    fallback_chunk: dict[str, Any],
) -> ChunkMetaData:
    if not isinstance(metadata, dict):
        start = int(fallback_chunk.get("start", 0))
        end = int(fallback_chunk.get("end", 0))
        image_names = fallback_chunk.get("image_names", [])
        return ChunkMetaData(
            entry_index=int(fallback_chunk.get("entry_index", 0)),
            start=start,
            end=end,
            char_count=end - start,
            image_count=len(image_names) if isinstance(image_names, list) else 0,
            chunking_strategy="legacy",
            heading="",
        )

    return ChunkMetaData(
        entry_index=int(metadata.get("entry_index", fallback_chunk.get("entry_index", 0))),
        start=int(metadata.get("start", fallback_chunk.get("start", 0))),
        end=int(metadata.get("end", fallback_chunk.get("end", 0))),
        char_count=int(metadata.get("char_count", 0)),
        image_count=int(metadata.get("image_count", 0)),
        chunking_strategy=str(metadata.get("chunking_strategy", "")),
        heading=str(metadata.get("heading", "")),
    )


def upgrade_cache_files(cache_dir: Path = RAG_CACHE_DIR) -> int:
    upgraded = 0
    if not cache_dir.exists():
        return upgraded

    for cache_path in cache_dir.glob("*.pkl"):
        try:
            with cache_path.open("rb") as file:
                payload = pickle.load(file)
        except (OSError, pickle.PickleError, EOFError, AttributeError):
            continue
        if not isinstance(payload, dict):
            continue

        chunks = payload.get("chunks", [])
        if chunks and isinstance(chunks[0], dict):
            continue

        payload["chunks"] = [
            serialize_chunk(deserialize_chunk(chunk))
            for chunk in chunks
        ]
        try:
            with cache_path.open("wb") as file:
                pickle.dump(payload, file)
        except OSError:
            continue
        upgraded += 1

    return upgraded


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def product_name_from_manual(manual_path: Path) -> str:
    return manual_path.stem.removesuffix(MANUAL_SUFFIX)


def manual_files(manual_dir: Path = MANUAL_DIR) -> list[Path]:
    if not manual_dir.exists():
        return []
    return sorted(
        path
        for path in manual_dir.glob("*.txt")
        if path.is_file()
    )


def warmup_targets(manual_dir: Path = MANUAL_DIR) -> list[tuple[Path, str | None]]:
    targets: list[tuple[Path, str | None]] = []
    for manual_path in manual_files(manual_dir):
        manual_product_name = product_name_from_manual(manual_path)
        if manual_product_name == ENGLISH_SUMMARY_PRODUCT:
            targets.extend(
                (manual_path, product_name)
                for product_name in english_summary_product_names()
            )
        else:
            targets.append((manual_path, None))
    return targets


def safe_cache_name(value: str) -> str:
    safe = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value, flags=re.UNICODE).strip("._")
    return safe or "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_EMBEDDING_MODEL_DIR)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=RAG_CACHE_DIR)
    parser.add_argument("--manual", type=Path)
    parser.add_argument("--product-name")
    parser.add_argument("--question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbor-count", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-images-per-chunk", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--warmup-all", action="store_true")
    parser.add_argument("--upgrade-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_model:
        model_dir = download_embedding_model(args.model_name, args.model_dir)
        print(f"downloaded to {model_dir}")
        return

    if args.upgrade_cache:
        upgraded = upgrade_cache_files(args.cache_dir)
        print(f"upgraded cache files={upgraded}")
        return

    rag = ManualRAG(
        embedding_model_dir=args.model_dir,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        max_images_per_chunk=args.max_images_per_chunk,
        device=args.device,
    )

    if args.warmup_all:
        warmed = []
        for manual_path, product_name in warmup_targets(MANUAL_DIR):
            index = rag.load_or_build_index(
                manual_path=manual_path,
                product_name=product_name,
                rebuild_cache=args.rebuild_cache,
            )
            item = {
                "manual_path": manual_path,
                "product_name": product_name,
                "chunk_count": len(index["chunks"]),
                "document_count": len(index["document_vectors"]),
            }
            warmed.append(item)
            display_product_name = product_name or "all"
            print(
                f"cached manual={item['manual_path'].name} "
                f"product={display_product_name} "
                f"chunks={item['chunk_count']} "
                f"documents={item['document_count']}",
                flush=True,
            )
        print(f"warmup finished, total={len(warmed)}")
        return

    if not args.manual:
        print("请提供 --manual，或使用 --download-model 下载模型。")
        return

    if args.dry_run:
        chunks = rag.build_chunks(args.manual, product_name=args.product_name)
        print(f"chunks={len(chunks)}")
        for index, chunk in enumerate(chunks[: args.top_k], start=1):
            print(f"#{index} entry={chunk.entry_index} span={chunk.start}:{chunk.end}")
            print(f"images={chunk.image_names}")
            print(f"image_paths={[str(path) for path in chunk.image_paths]}")
            print(chunk.text.replace("\n", " ")[:300])
            print()
        return

    if not args.question:
        print("请提供 --question，或使用 --dry-run 只检查切块。")
        return

    results = rag.retrieve(
        args.question,
        args.manual,
        product_name=args.product_name,
        top_k=args.top_k,
        neighbor_count=args.neighbor_count,
        rebuild_cache=args.rebuild_cache,
    )
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"#{index} score={result.score:.4f} entry={chunk.entry_index} span={chunk.start}:{chunk.end}")
        print(f"images={chunk.image_names}")
        print(f"image_paths={[str(path) for path in chunk.image_paths]}")
        print(chunk.text.replace("\n", " ")[:300])
        print()


if __name__ == "__main__":
    main()
