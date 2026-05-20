import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag import ManualRAG, RetrievalResult, chunk_key
from rag_vote_route_test import global_retrieve, load_all_indexes
from test_intent_rag_assisted import DEFAULT_IDS, EXPECTED, aggregate_candidates, load_questions


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RERANKER_DIR = BASE_DIR / "Qwen3-Reranker-0.6B" / "models" / "Qwen3-Reranker-0.6B"
DEFAULT_OUTPUT = Path("rerank_top3_compare.csv")


def normalize_text(text: str, max_chars: int = 1100) -> str:
    return " ".join(text.replace("<PIC>", " ").split())[:max_chars]


def format_instruction(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def label_chunk_indexes(indexes: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {index["label"]: index["chunks"] for index in indexes}


def expand_chunk_text(label: str, result: RetrievalResult, chunks_by_label: dict[str, list[Any]], neighbor_count: int) -> str:
    if neighbor_count <= 0:
        return result.chunk.text
    chunks = chunks_by_label.get(label, [])
    positions = {chunk_key(chunk): index for index, chunk in enumerate(chunks)}
    center = positions.get(chunk_key(result.chunk))
    if center is None:
        return result.chunk.text
    texts = []
    for offset in range(-neighbor_count, neighbor_count + 1):
        index = center + offset
        if index < 0 or index >= len(chunks):
            continue
        chunk = chunks[index]
        if chunk.entry_index != result.chunk.entry_index:
            continue
        texts.append(chunk.text)
    return "\n".join(texts) if texts else result.chunk.text


def aggregate_candidates_with_results(
    results: list[tuple[str, RetrievalResult]],
    chunks_by_label: dict[str, list[Any]],
    manual_count: int,
    snippets_per_manual: int,
    neighbor_count: int,
    score_top_n: int = 3,
) -> list[dict[str, Any]]:
    by_label: dict[str, list[RetrievalResult]] = {}
    for label, result in results:
        by_label.setdefault(label, []).append(result)

    ranked = []
    for label, label_results in by_label.items():
        sorted_results = sorted(label_results, key=lambda item: item.score, reverse=True)
        top_scores = [item.score for item in sorted_results[:score_top_n]]
        if not top_scores:
            continue
        ranked.append(
            {
                "label": label,
                "max_score": max(top_scores),
                "avg_score": sum(top_scores) / len(top_scores),
                "results": sorted_results,
            }
        )

    ranked.sort(key=lambda item: (-item["avg_score"], -item["max_score"], item["label"]))
    candidates = []
    for item in ranked[:manual_count]:
        snippets = []
        for result in item["results"][:snippets_per_manual]:
            snippets.append(
                {
                    "score": result.score,
                    "text": expand_chunk_text(item["label"], result, chunks_by_label, neighbor_count),
                    "entry_index": result.chunk.entry_index,
                    "start": result.chunk.start,
                    "end": result.chunk.end,
                }
            )
        candidates.append(
            {
                "label": item["label"],
                "max_score": item["max_score"],
                "avg_score": item["avg_score"],
                "snippets": snippets,
            }
        )
    return candidates


class QwenReranker:
    def __init__(
        self,
        model_dir: Path,
        max_length: int = 1024,
        batch_size: int = 2,
    ) -> None:
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

    def score_pairs(self, pairs: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            scores.extend(self._score_batch(batch))
        return scores

    @torch.no_grad()
    def _score_batch(self, pairs: list[str]) -> list[float]:
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
        logits = torch.stack([false_vector, true_vector], dim=1)
        return torch.nn.functional.log_softmax(logits, dim=1)[:, 1].exp().tolist()


def rerank_candidates(
    reranker: QwenReranker,
    question: str,
    candidates: list[dict[str, Any]],
    instruction: str,
) -> tuple[list[str], str]:
    pairs = []
    snippet_rows = []
    for candidate in candidates:
        for snippet in candidate["snippets"]:
            text = normalize_text(snippet["text"])
            pairs.append(format_instruction(instruction, question, text))
            snippet_rows.append(
                {
                    "label": candidate["label"],
                    "embedding_score": float(snippet["score"]),
                    "text": text,
                }
            )

    if not pairs:
        return [], ""

    scores = reranker.score_pairs(pairs)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for item, score in zip(snippet_rows, scores):
        item["rerank_score"] = score
        by_label.setdefault(item["label"], []).append(item)

    label_scores = []
    for label, items in by_label.items():
        sorted_items = sorted(items, key=lambda item: item["rerank_score"], reverse=True)
        top_scores = [item["rerank_score"] for item in sorted_items[:2]]
        label_scores.append(
            {
                "label": label,
                "max_score": max(top_scores),
                "avg_score": sum(top_scores) / len(top_scores),
                "items": sorted_items,
            }
        )
    label_scores.sort(key=lambda item: (-item["avg_score"], -item["max_score"], item["label"]))
    top_labels = [item["label"] for item in label_scores[:3]]
    detail = " | ".join(
        f"{item['label']}:avg{item['avg_score']:.4f}/max{item['max_score']:.4f}"
        for item in label_scores[:10]
    )
    return top_labels, detail


def hit(expected: str, labels: list[str]) -> str:
    return str(expected in labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("question_public.csv"))
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reranker-dir", type=Path, default=DEFAULT_RERANKER_DIR)
    parser.add_argument("--global-top-k", type=int, default=80)
    parser.add_argument("--candidate-manuals", type=int, default=10)
    parser.add_argument("--snippets-per-manual", type=int, default=2)
    parser.add_argument("--neighbor-count", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--instruction",
        default="Given a product manual routing question, retrieve manual passages that identify the correct product manual and can answer the user's question.",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions, [str(item) for item in args.ids])
    rag = ManualRAG()
    indexes = load_all_indexes(rag)
    chunks_by_label = label_chunk_indexes(indexes)
    reranker = QwenReranker(args.reranker_dir, max_length=args.max_length, batch_size=args.batch_size)

    fieldnames = [
        "id",
        "question",
        "expected",
        "embedding_top3",
        "rerank_top3",
        "embedding_hit",
        "rerank_hit",
        "embedding_candidate10",
        "expected_in_candidate10",
        "neighbor_count",
        "rerank_detail",
    ]
    rows = []
    for index, row in enumerate(questions, start=1):
        results = global_retrieve(rag, indexes, row["question"], args.global_top_k)
        embedding_top3_candidates = aggregate_candidates(results, manual_count=3, snippets_per_manual=2)
        candidate10 = aggregate_candidates(
            results,
            manual_count=args.candidate_manuals,
            snippets_per_manual=args.snippets_per_manual,
        )
        rerank_candidate10 = aggregate_candidates_with_results(
            results,
            chunks_by_label=chunks_by_label,
            manual_count=args.candidate_manuals,
            snippets_per_manual=args.snippets_per_manual,
            neighbor_count=args.neighbor_count,
        )
        embedding_top3 = [candidate["label"] for candidate in embedding_top3_candidates]
        rerank_top3, rerank_detail = rerank_candidates(reranker, row["question"], rerank_candidate10, args.instruction)
        candidate10_labels = [candidate["label"] for candidate in candidate10]
        expected = EXPECTED.get(row["id"], "")

        result_row = {
            "id": row["id"],
            "question": row["question"],
            "expected": expected,
            "embedding_top3": " | ".join(embedding_top3),
            "rerank_top3": " | ".join(rerank_top3),
            "embedding_hit": hit(expected, embedding_top3),
            "rerank_hit": hit(expected, rerank_top3),
            "embedding_candidate10": " | ".join(candidate10_labels),
            "expected_in_candidate10": str(expected in candidate10_labels),
            "neighbor_count": str(args.neighbor_count),
            "rerank_detail": rerank_detail,
        }
        rows.append(result_row)
        print(
            f"[{index}/{len(questions)}] id={row['id']} "
            f"emb={result_row['embedding_hit']} rerank={result_row['rerank_hit']}",
            flush=True,
        )

    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved to {args.output}")
    for key in ["embedding_hit", "rerank_hit", "expected_in_candidate10"]:
        count = sum(row[key] == "True" for row in rows)
        print(f"{key}={count}/{len(rows)}")


if __name__ == "__main__":
    main()
