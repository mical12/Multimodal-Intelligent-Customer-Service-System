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

DEFAULT_EMBEDDING_MODEL_DIR = BASE_DIR / "Qwen3-VL-Embedding-2B"
DEFAULT_RERANKER_MODEL_DIR = BASE_DIR / "Qwen3-Reranker-0.6B" / "models" / "Qwen3-Reranker-0.6B"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_RERANKER_BATCH_SIZE = 2
DEFAULT_RERANKER_MAX_LENGTH = 1024
MIN_STANDALONE_SECTION_CHARS = 100
DEFAULT_QUERY_PROMPT = "Retrieve relevant product manual passages for the user's question."
DEFAULT_RERANK_INSTRUCTION = (
    "Given a product manual question, retrieve the manual passage that contains "
    "the information needed to answer the user's question."
)
ENGLISH_SUMMARY_PRODUCT = "英文汇总"
MANUAL_SUFFIX = "手册"
PIC_TOKEN = "<PIC>"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CACHE_VERSION = 3
HEADING_PATTERN = re.compile(r"(?<!\S)#\s+")
TOC_DOT_LEADER_PATTERN = re.compile(r"\.{6,}")
SENTENCE_END_PATTERN = re.compile(r"[。！？!?]|(?<!\d)\.(?!\d)")


BM25_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*|[\u4e00-\u9fff]")


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
    parent_start: int | None = None
    parent_end: int | None = None
    child_index: int = 0
    child_count: int = 1


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


class QwenReranker:
    def __init__(
        self,
        model_dir: Path = DEFAULT_RERANKER_MODEL_DIR,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    ) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(f"reranker model not found: {model_dir}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            padding_side="left",
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.float32,
        ).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on '
            'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def score(
        self,
        question: str,
        documents: list[str],
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
    ) -> list[float]:
        pairs = [
            f"<Instruct>: {instruction}\n<Query>: {question}\n<Document>: {normalize_rerank_text(document)}"
            for document in documents
        ]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            scores.extend(self._score_batch(pairs[start : start + self.batch_size]))
        return scores

    def _score_batch(self, pairs: list[str]) -> list[float]:
        with self.torch.no_grad():
            budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
            inputs = self.tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=budget,
            )
            for index, input_ids in enumerate(inputs["input_ids"]):
                inputs["input_ids"][index] = self.prefix_tokens + input_ids + self.suffix_tokens
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
            outputs = self.model(**inputs).logits[:, -1, :]
            true_vector = outputs[:, self.token_true_id]
            false_vector = outputs[:, self.token_false_id]
            logits = self.torch.stack([false_vector, true_vector], dim=1)
            return self.torch.nn.functional.log_softmax(logits, dim=1)[:, 1].exp().tolist()


_default_rag: "ManualRAG | None" = None


def get_default_rag() -> "ManualRAG":
    """Return one reusable RAG instance so the embedding model is loaded once."""
    global _default_rag
    if _default_rag is None:
        _default_rag = ManualRAG()
    return _default_rag


class ManualRAG:
    """Vector-only RAG over one known manual using Qwen3-VL-Embedding."""

    def __init__(
        self,
        embedding_model_dir: Path = DEFAULT_EMBEDDING_MODEL_DIR,
        image_dir: Path = IMAGE_DIR,
        cache_dir: Path = RAG_CACHE_DIR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_images_per_chunk: int = 0,
        query_prompt: str = DEFAULT_QUERY_PROMPT,
        device: str = "cpu",
    ):
        if max_images_per_chunk < 0:
            raise ValueError("max_images_per_chunk 不能小于 0。")

        self.embedding_model_dir = embedding_model_dir
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.max_images_per_chunk = max_images_per_chunk
        self.query_prompt = query_prompt
        self.device = device
        self._embedder: Any | None = None
        self._reranker: QwenReranker | None = None

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

        results = sorted(best_by_chunk.values(), key=lambda item: item.score, reverse=True)[:top_k]
        if neighbor_count > 0:
            return expand_with_adjacent_child_chunks(results, chunks, neighbor_count)
        return results

    def retrieve_with_bm25_rerank(
        self,
        question: str,
        manual_path: Path,
        product_name: str | None = None,
        *,
        embedding_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 4,
        neighbor_count: int = 0,
        reranker_model_dir: Path = DEFAULT_RERANKER_MODEL_DIR,
        reranker_max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        reranker_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    ) -> list[RetrievalResult]:
        embedding_results = self.retrieve(
            question=question,
            manual_path=manual_path,
            product_name=product_name,
            top_k=embedding_top_k,
        )
        bm25_results = self.retrieve_bm25(
            question=question,
            manual_path=manual_path,
            product_name=product_name,
            top_k=bm25_top_k,
        )
        candidates = merge_retrieval_results(embedding_results, bm25_results)
        if not candidates:
            return []

        reranker = self._get_reranker(
            model_dir=reranker_model_dir,
            max_length=reranker_max_length,
            batch_size=reranker_batch_size,
        )
        scores = reranker.score(question, [result.chunk.text for result in candidates])
        reranked = sorted(
            zip(candidates, scores, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
        results = [
            RetrievalResult(chunk=result.chunk, score=float(score))
            for result, score in reranked[:final_top_k]
        ]
        if neighbor_count > 0:
            index = self.load_or_build_index(manual_path=manual_path, product_name=product_name)
            chunks: list[ManualChunk] = index["chunks"]
            return expand_with_adjacent_child_chunks(results, chunks, neighbor_count)
        return results

    def retrieve_bm25(
        self,
        question: str,
        manual_path: Path,
        product_name: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        index = self.load_or_build_index(manual_path=manual_path, product_name=product_name)
        chunks: list[ManualChunk] = index["chunks"]
        if not chunks:
            return []

        tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        query_tokens = tokenize_for_bm25(question)
        if not query_tokens:
            return []

        doc_freq: dict[str, int] = {}
        for tokens in tokenized_chunks:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        avgdl = sum(len(tokens) for tokens in tokenized_chunks) / max(len(tokenized_chunks), 1)
        total_docs = len(tokenized_chunks)
        query_unique = set(query_tokens)
        k1 = 1.5
        b = 0.75
        results: list[RetrievalResult] = []
        for chunk, tokens in zip(chunks, tokenized_chunks, strict=False):
            if not tokens:
                continue
            term_freq: dict[str, int] = {}
            for token in tokens:
                if token in query_unique:
                    term_freq[token] = term_freq.get(token, 0) + 1
            if not term_freq:
                continue

            score = 0.0
            length_norm = k1 * (1 - b + b * len(tokens) / max(avgdl, 1))
            for token, freq in term_freq.items():
                df = doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                score += idf * (freq * (k1 + 1)) / (freq + length_norm)
            if score > 0:
                results.append(RetrievalResult(chunk=chunk, score=score))

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def load_or_build_index(
        self,
        manual_path: Path,
        product_name: str | None = None,
        rebuild_cache: bool = False,
    ) -> dict[str, Any]:
        product_name = cache_product_name(manual_path, product_name)
        entries = load_manual_entries(manual_path)
        entry_indexes = selected_entry_indexes(manual_path, product_name, len(entries))
        chunks = split_manual_entries(
            entries,
            entry_indexes=entry_indexes,
            image_dir=self.image_dir,
            chunk_size=self.chunk_size,
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
        product_name = cache_product_name(manual_path, product_name)
        entries = load_manual_entries(manual_path)
        return split_manual_entries(
            entries,
            entry_indexes=selected_entry_indexes(manual_path, product_name, len(entries)),
            image_dir=self.image_dir,
            chunk_size=self.chunk_size,
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
                "请通过 --model-dir 指定正确的本地模型目录。"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 sentence-transformers，请先安装：pip install sentence-transformers"
            ) from exc

        return SentenceTransformer(str(self.embedding_model_dir), device=self.device)

    def _get_reranker(
        self,
        model_dir: Path,
        max_length: int,
        batch_size: int,
    ) -> QwenReranker:
        if self._reranker is None:
            self._reranker = QwenReranker(
                model_dir=model_dir,
                max_length=max_length,
                batch_size=batch_size,
            )
        return self._reranker

    def _cache_path(self, manual_path: Path, product_name: str | None) -> Path:
        product_name = cache_product_name(manual_path, product_name)
        product_part = product_name or "all"
        key = "|".join(
            [
                str(manual_path.resolve()),
                product_part,
                str(self.chunk_size),
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
        product_name = cache_product_name(manual_path, product_name)
        stat = manual_path.stat()
        return {
            "cache_version": CACHE_VERSION,
            "manual_path": str(manual_path.resolve()),
            "manual_mtime_ns": stat.st_mtime_ns,
            "manual_size": stat.st_size,
            "product_name": product_name,
            "entry_indexes": entry_indexes,
            "chunk_size": self.chunk_size,
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


def cache_product_name(manual_path: Path, product_name: str | None) -> str | None:
    if product_name_from_manual(manual_path) == ENGLISH_SUMMARY_PRODUCT:
        return product_name
    return None


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

        for span in section_chunk_spans(content, target_chars=chunk_size):
            start, end, strategy, parent_start, parent_end, child_index, child_count = span
            chunks.append(
                build_manual_chunk(
                    content=content,
                    entry_index=entry_index,
                    start=start,
                    end=end,
                    strategy=strategy,
                    parent_start=parent_start,
                    parent_end=parent_end,
                    child_index=child_index,
                    child_count=child_count,
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
    parent_start: int,
    parent_end: int,
    child_index: int,
    child_count: int,
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
        parent_start=parent_start,
        parent_end=parent_end,
        child_index=child_index,
        child_count=child_count,
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


ChunkSpan = tuple[int, int, str, int, int, int, int]


def make_chunk_span(
    start: int,
    end: int,
    strategy: str,
    parent_start: int | None = None,
    parent_end: int | None = None,
    child_index: int = 0,
    child_count: int = 1,
) -> ChunkSpan:
    return (
        start,
        end,
        strategy,
        start if parent_start is None else parent_start,
        end if parent_end is None else parent_end,
        child_index,
        child_count,
    )


def section_chunk_spans(content: str, target_chars: int = DEFAULT_CHUNK_SIZE) -> list[ChunkSpan]:
    raw_spans = raw_section_spans(content)
    raw_spans = merge_tiny_section_spans(raw_spans, content, min_chars=MIN_STANDALONE_SECTION_CHARS)
    raw_spans = drop_table_of_contents_spans(raw_spans, content)
    raw_spans = merge_tiny_section_spans(raw_spans, content, min_chars=MIN_STANDALONE_SECTION_CHARS)
    clustered_spans = cluster_section_spans(raw_spans, target_chars=target_chars)
    return split_long_spans_by_sentence(clustered_spans, content, target_chars=target_chars)


def raw_section_spans(content: str) -> list[ChunkSpan]:
    starts = [match.start() for match in HEADING_PATTERN.finditer(content)]
    if not starts:
        return [make_chunk_span(0, len(content), "section_whole")]
    if starts[0] > 0:
        starts = [0] + starts

    spans: list[ChunkSpan] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(content)
        while start < end and content[start].isspace():
            start += 1
        while end > start and content[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append(make_chunk_span(start, end, "section"))
    return spans


def drop_table_of_contents_spans(
    spans: list[ChunkSpan],
    content: str,
) -> list[ChunkSpan]:
    return [
        span
        for span in spans
        for start, end, *_ in [span]
        if not is_table_of_contents_text(content[start:end])
    ]


def is_table_of_contents_text(text: str) -> bool:
    dot_leader_count = len(TOC_DOT_LEADER_PATTERN.findall(text))
    if dot_leader_count < 8:
        return False

    normalized = " ".join(text.casefold().split())
    return (
        "contents" in normalized
        or "table of contents" in normalized
        or "目录" in normalized
        or dot_leader_count >= 15
    )


def merge_tiny_section_spans(
    spans: list[ChunkSpan],
    content: str,
    min_chars: int,
) -> list[ChunkSpan]:
    if len(spans) <= 1:
        return spans

    merged: list[ChunkSpan] = []
    pending_start: int | None = None

    for index, span in enumerate(spans):
        start, end, strategy, *_ = span
        if pending_start is not None:
            start = pending_start
            strategy = "section_tiny_merged"
            pending_start = None

        text_length = len(" ".join(content[start:end].replace(PIC_TOKEN, " ").split()))
        if text_length < min_chars:
            if index == len(spans) - 1:
                if merged:
                    previous_start = merged[-1][0]
                    merged[-1] = make_chunk_span(previous_start, end, "section_tiny_merged")
                else:
                    merged.append(make_chunk_span(start, end, strategy))
            else:
                pending_start = start
            continue

        merged.append(make_chunk_span(start, end, strategy))

    return merged


def cluster_section_spans(
    spans: list[ChunkSpan],
    target_chars: int,
) -> list[ChunkSpan]:
    if not spans:
        return []

    clustered: list[ChunkSpan] = []
    cluster_start, cluster_end, cluster_strategy, *_ = spans[0]
    for span in spans[1:]:
        start, end, strategy, *_ = span
        cluster_length = cluster_end - cluster_start
        merged_length = end - cluster_start
        if cluster_length < target_chars and merged_length <= target_chars:
            cluster_end = end
            cluster_strategy = "section_cluster"
            continue

        clustered.append(make_chunk_span(cluster_start, cluster_end, cluster_strategy))
        cluster_start, cluster_end, cluster_strategy = start, end, strategy

    clustered.append(make_chunk_span(cluster_start, cluster_end, cluster_strategy))
    return clustered


def split_long_spans_by_sentence(
    spans: list[ChunkSpan],
    content: str,
    target_chars: int,
) -> list[ChunkSpan]:
    split_spans: list[ChunkSpan] = []
    for span in spans:
        start, end, strategy, *_ = span
        if end - start <= target_chars:
            split_spans.append(make_chunk_span(start, end, strategy))
            continue

        sentence_spans = sentence_spans_for_range(content, start, end)
        if len(sentence_spans) <= 1:
            split_spans.append(make_chunk_span(start, end, strategy))
            continue

        parent_start, parent_end = start, end
        child_spans: list[tuple[int, int]] = []
        chunk_start, chunk_end = sentence_spans[0]
        for sentence_start, sentence_end in sentence_spans[1:]:
            merged_length = sentence_end - chunk_start
            if chunk_end > chunk_start and merged_length > target_chars:
                child_spans.append((chunk_start, chunk_end))
                chunk_start, chunk_end = sentence_start, sentence_end
            else:
                chunk_end = sentence_end

        child_spans.append((chunk_start, chunk_end))
        child_spans = merge_short_child_ranges(child_spans, content, MIN_STANDALONE_SECTION_CHARS)
        for child_index, (child_start, child_end) in enumerate(child_spans):
            split_spans.append(
                make_chunk_span(
                    child_start,
                    child_end,
                    "section_sentence_split",
                    parent_start=parent_start,
                    parent_end=parent_end,
                    child_index=child_index,
                    child_count=len(child_spans),
                )
            )

    return split_oversized_spans_by_whitespace(split_spans, content, target_chars=target_chars)


def split_oversized_spans_by_whitespace(
    spans: list[ChunkSpan],
    content: str,
    target_chars: int,
) -> list[ChunkSpan]:
    split_spans: list[ChunkSpan] = []
    for span in spans:
        start, end, strategy, parent_start, parent_end, _, _ = span
        if end - start <= target_chars:
            split_spans.append(span)
            continue

        child_spans: list[tuple[int, int]] = []
        chunk_start = start
        while end - chunk_start > target_chars:
            split_at = content.rfind(" ", chunk_start, chunk_start + target_chars + 1)
            if split_at <= chunk_start:
                split_at = chunk_start + target_chars
            while split_at > chunk_start and content[split_at - 1].isspace():
                split_at -= 1
            if split_at <= chunk_start:
                split_at = min(chunk_start + target_chars, end)
            child_spans.append((chunk_start, split_at))
            chunk_start = split_at
            while chunk_start < end and content[chunk_start].isspace():
                chunk_start += 1

        if chunk_start < end:
            child_spans.append((chunk_start, end))

        child_spans = merge_short_child_ranges(child_spans, content, MIN_STANDALONE_SECTION_CHARS)
        for child_index, (child_start, child_end) in enumerate(child_spans):
            split_spans.append(
                make_chunk_span(
                    child_start,
                    child_end,
                    "section_word_split",
                    parent_start=parent_start,
                    parent_end=parent_end,
                    child_index=child_index,
                    child_count=len(child_spans),
                )
            )

    return split_spans


def merge_short_child_ranges(
    ranges: list[tuple[int, int]],
    content: str,
    min_chars: int,
) -> list[tuple[int, int]]:
    if len(ranges) <= 1:
        return ranges

    merged: list[tuple[int, int]] = []
    index = 0
    while index < len(ranges):
        start, end = ranges[index]
        text_length = len(" ".join(content[start:end].replace(PIC_TOKEN, " ").split()))
        if text_length >= min_chars:
            merged.append((start, end))
            index += 1
            continue

        if index + 1 < len(ranges):
            _, next_end = ranges[index + 1]
            merged.append((start, next_end))
            index += 2
            continue

        if merged:
            previous_start, _ = merged[-1]
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
        index += 1

    return merged


def sentence_spans_for_range(content: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sentence_start = start
    for match in SENTENCE_END_PATTERN.finditer(content, start, end):
        sentence_end = match.end()
        while sentence_end < end and content[sentence_end].isspace():
            sentence_end += 1
        if sentence_end > sentence_start:
            spans.append((sentence_start, sentence_end))
        sentence_start = sentence_end

    if sentence_start < end:
        spans.append((sentence_start, end))

    return [
        (span_start, span_end)
        for span_start, span_end in spans
        if content[span_start:span_end].strip()
    ]


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
        "parent_start": metadata.parent_start,
        "parent_end": metadata.parent_end,
        "child_index": metadata.child_index,
        "child_count": metadata.child_count,
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
            parent_start=start,
            parent_end=end,
            child_index=0,
            child_count=1,
        )

    return ChunkMetaData(
        entry_index=int(metadata.get("entry_index", fallback_chunk.get("entry_index", 0))),
        start=int(metadata.get("start", fallback_chunk.get("start", 0))),
        end=int(metadata.get("end", fallback_chunk.get("end", 0))),
        char_count=int(metadata.get("char_count", 0)),
        image_count=int(metadata.get("image_count", 0)),
        chunking_strategy=str(metadata.get("chunking_strategy", "")),
        heading=str(metadata.get("heading", "")),
        parent_start=int(metadata.get("parent_start", metadata.get("start", fallback_chunk.get("start", 0)))),
        parent_end=int(metadata.get("parent_end", metadata.get("end", fallback_chunk.get("end", 0)))),
        child_index=int(metadata.get("child_index", 0)),
        child_count=int(metadata.get("child_count", 1)),
    )


def normalize_rerank_text(text: str, max_chars: int = 1800) -> str:
    return " ".join(text.replace(PIC_TOKEN, " ").split())[:max_chars]


def tokenize_for_bm25(text: str) -> list[str]:
    return [
        token.lower()
        for token in BM25_TOKEN_PATTERN.findall(text.replace(PIC_TOKEN, " "))
    ]


def retrieval_key(result: RetrievalResult) -> tuple[int, int, int]:
    chunk = result.chunk
    return (chunk.entry_index, chunk.start, chunk.end)


def merge_retrieval_results(
    primary: list[RetrievalResult],
    secondary: list[RetrievalResult],
) -> list[RetrievalResult]:
    merged: dict[tuple[int, int, int], RetrievalResult] = {}
    for result in primary:
        merged[retrieval_key(result)] = result
    for result in secondary:
        merged.setdefault(retrieval_key(result), result)
    return list(merged.values())


def expand_with_adjacent_child_chunks(
    results: list[RetrievalResult],
    chunks: list[ManualChunk],
    neighbor_count: int,
) -> list[RetrievalResult]:
    if neighbor_count <= 0 or not results:
        return results

    child_lookup: dict[tuple[int, int, int, int], ManualChunk] = {}
    for chunk in chunks:
        metadata = chunk.metadata
        if metadata is None:
            continue
        parent_start = metadata.parent_start if metadata.parent_start is not None else chunk.start
        parent_end = metadata.parent_end if metadata.parent_end is not None else chunk.end
        child_lookup[
            (
                chunk.entry_index,
                parent_start,
                parent_end,
                metadata.child_index,
            )
        ] = chunk

    expanded: list[RetrievalResult] = []
    seen: set[tuple[int, int, int]] = set()
    for result in results:
        group = [result]
        metadata = result.chunk.metadata
        if metadata is not None and metadata.child_count > 1:
            parent_start = metadata.parent_start if metadata.parent_start is not None else result.chunk.start
            parent_end = metadata.parent_end if metadata.parent_end is not None else result.chunk.end
            start_index = max(0, metadata.child_index - neighbor_count)
            end_index = min(metadata.child_count - 1, metadata.child_index + neighbor_count)
            for child_index in range(start_index, end_index + 1):
                neighbor = child_lookup.get(
                    (
                        result.chunk.entry_index,
                        parent_start,
                        parent_end,
                        child_index,
                    )
                )
                if neighbor is None:
                    continue
                group.append(RetrievalResult(chunk=neighbor, score=result.score))

        group.sort(key=lambda item: (item.chunk.entry_index, item.chunk.start, item.chunk.end))
        for item in group:
            key = retrieval_key(item)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(item)

    return expanded


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
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_EMBEDDING_MODEL_DIR)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=RAG_CACHE_DIR)
    parser.add_argument("--manual", type=Path)
    parser.add_argument("--product-name")
    parser.add_argument("--question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-images-per-chunk", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--warmup-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rag = ManualRAG(
        embedding_model_dir=args.model_dir,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
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
        print("请提供 --manual，或使用 --warmup-all 预热全部手册。")
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
