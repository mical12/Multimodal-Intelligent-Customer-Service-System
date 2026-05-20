import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from rag import (
    MANUAL_DIR,
    ManualRAG,
    RetrievalResult,
    cosine_similarity,
    product_name_from_manual,
    warmup_targets,
)


BASE_DIR = Path(__file__).resolve().parent
QUESTION_PATH = BASE_DIR / "question_public.csv"
INTENT_LOG_PATH = BASE_DIR / "intent_log.jsonl"
OUTPUT_PATH = BASE_DIR / "rag_vote_route_result.csv"


def read_questions(path: Path = QUESTION_PATH) -> list[tuple[int, str]]:
    questions: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if not row:
                continue
            try:
                qid = int(row[0])
            except ValueError:
                continue
            questions.append((qid, row[1]))
    return questions


def target_label(manual_path: Path, product_name: str | None) -> str:
    manual_name = product_name_from_manual(manual_path)
    return product_name or manual_name


def load_latest_intent_routes(path: Path = INTENT_LOG_PATH) -> dict[int, dict[str, str]]:
    routes: dict[int, dict[str, str]] = {}
    if not path.exists():
        return routes

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = str(item.get("user_id") or "")
            if not user_id.startswith("test_"):
                continue
            try:
                qid = int(user_id.removeprefix("test_"))
            except ValueError:
                continue
            routes[qid] = {
                "intent_agent_type": str(item.get("agent_type") or ""),
                "intent_product_name": str(item.get("product_name") or ""),
                "intent_reason": str(item.get("reason") or ""),
            }
    return routes


def load_all_indexes(rag: ManualRAG) -> list[dict]:
    indexes: list[dict] = []
    for manual_path, product_name in warmup_targets(MANUAL_DIR):
        index = rag.load_or_build_index(manual_path, product_name=product_name)
        indexes.append(
            {
                "manual_path": manual_path,
                "product_name": product_name,
                "label": target_label(manual_path, product_name),
                "chunks": index["chunks"],
                "document_chunk_indexes": index["document_chunk_indexes"],
                "document_vectors": index["document_vectors"],
            }
        )
    return indexes


def global_retrieve(
    rag: ManualRAG,
    indexes: list[dict],
    question: str,
    top_k: int,
) -> list[tuple[str, RetrievalResult]]:
    question_vector = rag._encode_query(question)
    best: list[tuple[str, RetrievalResult]] = []

    for index in indexes:
        best_by_chunk: dict[int, RetrievalResult] = {}
        for chunk_index, document_vector in zip(
            index["document_chunk_indexes"],
            index["document_vectors"],
        ):
            chunk = index["chunks"][chunk_index]
            score = cosine_similarity(question_vector, document_vector)
            current = best_by_chunk.get(chunk_index)
            if current is None or score > current.score:
                best_by_chunk[chunk_index] = RetrievalResult(chunk=chunk, score=score)
        best.extend((index["label"], result) for result in best_by_chunk.values())

    return sorted(best, key=lambda item: item[1].score, reverse=True)[:top_k]


def vote_label(results: list[tuple[str, RetrievalResult]]) -> tuple[str, str, str]:
    counts = Counter(label for label, _ in results)
    score_sums = Counter()
    for label, result in results:
        score_sums[label] += result.score

    if not counts:
        return "", "", ""

    count_winner = max(counts, key=lambda label: (counts[label], score_sums[label]))
    score_winner = max(score_sums, key=lambda label: (score_sums[label], counts[label]))
    distribution = "; ".join(
        f"{label}:{counts[label]}/{score_sums[label]:.3f}"
        for label in sorted(counts, key=lambda label: (-counts[label], -score_sums[label], label))
    )
    return count_winner, score_winner, distribution


def aggregate_label(
    results: list[tuple[str, RetrievalResult]],
    per_label_top_n: int = 3,
) -> tuple[str, str, str]:
    by_label: dict[str, list[float]] = {}
    for label, result in results:
        by_label.setdefault(label, []).append(result.score)

    if not by_label:
        return "", "", ""

    label_scores = {
        label: sorted(scores, reverse=True)[:per_label_top_n]
        for label, scores in by_label.items()
    }
    avg_scores = {
        label: sum(scores) / len(scores)
        for label, scores in label_scores.items()
        if scores
    }
    max_scores = {
        label: scores[0]
        for label, scores in label_scores.items()
        if scores
    }
    avg_winner = max(avg_scores, key=lambda label: (avg_scores[label], max_scores[label]))
    max_winner = max(max_scores, key=lambda label: (max_scores[label], avg_scores[label]))
    distribution = "; ".join(
        f"{label}:max{max_scores[label]:.3f}/avg{avg_scores[label]:.3f}/n{len(label_scores[label])}"
        for label in sorted(avg_scores, key=lambda label: (-avg_scores[label], -max_scores[label], label))
    )
    return avg_winner, max_winner, distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-id", type=int)
    parser.add_argument("--ids", nargs="*", type=int)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    questions = read_questions()
    if args.ids:
        wanted_ids = set(args.ids)
        questions = [(qid, question) for qid, question in questions if qid in wanted_ids]
    if args.start_id is not None:
        questions = [(qid, question) for qid, question in questions if qid >= args.start_id]
    if args.limit:
        questions = questions[: args.limit]

    rag = ManualRAG()
    indexes = load_all_indexes(rag)
    intent_routes = load_latest_intent_routes()

    fieldnames = [
            "id",
            "question",
            "intent_agent_type",
            "intent_product_name",
                "rag_vote_by_count",
                "rag_vote_by_score",
                "rag_top3_avg",
                "rag_top1_max",
                "same_as_intent_count",
                "same_as_intent_score",
                "same_as_intent_top3_avg",
                "same_as_intent_top1_max",
                "top10_distribution",
                "top3_aggregate_distribution",
                "top10_items",
    ]
    mode = "a" if args.append else "w"
    write_header = not args.append or not args.output.exists()
    rows = []
    with args.output.open(mode, encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for qid, question in questions:
            # Pull more than the final top10 so per-manual top3 has enough candidates.
            results = global_retrieve(rag, indexes, question, max(args.top_k, 60))
            display_results = results[: args.top_k]
            count_winner, score_winner, distribution = vote_label(results)
            avg3_winner, max_winner, aggregate_distribution = aggregate_label(results, per_label_top_n=3)
            top_items = " | ".join(
                f"{rank}.{label}:{result.score:.4f}:entry{result.chunk.entry_index}"
                for rank, (label, result) in enumerate(display_results, start=1)
            )
            intent = intent_routes.get(qid, {})
            intent_product = intent.get("intent_product_name", "")
            row = {
                    "id": qid,
                    "question": question,
                    "intent_agent_type": intent.get("intent_agent_type", ""),
                    "intent_product_name": intent_product,
                    "rag_vote_by_count": count_winner,
                    "rag_vote_by_score": score_winner,
                    "rag_top3_avg": avg3_winner,
                    "rag_top1_max": max_winner,
                    "same_as_intent_count": str(bool(intent_product and intent_product == count_winner)),
                    "same_as_intent_score": str(bool(intent_product and intent_product == score_winner)),
                    "same_as_intent_top3_avg": str(bool(intent_product and intent_product == avg3_winner)),
                    "same_as_intent_top1_max": str(bool(intent_product and intent_product == max_winner)),
                    "top10_distribution": distribution,
                    "top3_aggregate_distribution": aggregate_distribution,
                    "top10_items": top_items,
            }
            rows.append(row)
            writer.writerow(row)
            file.flush()
            print(
                f"{qid}: intent={intent_product or '-'} "
                f"vote={count_winner or '-'} avg3={avg3_winner or '-'} max={max_winner or '-'}",
                flush=True,
            )

    compared = [row for row in rows if row["intent_product_name"]]
    same_count = sum(row["same_as_intent_count"] == "True" for row in compared)
    same_score = sum(row["same_as_intent_score"] == "True" for row in compared)
    same_avg3 = sum(row["same_as_intent_top3_avg"] == "True" for row in compared)
    same_max = sum(row["same_as_intent_top1_max"] == "True" for row in compared)
    print(f"written={args.output}")
    print(f"questions={len(rows)} compared_with_intent={len(compared)}")
    print(f"same_as_intent_count_vote={same_count}/{len(compared)}")
    print(f"same_as_intent_score_vote={same_score}/{len(compared)}")
    print(f"same_as_intent_top3_avg={same_avg3}/{len(compared)}")
    print(f"same_as_intent_top1_max={same_max}/{len(compared)}")


if __name__ == "__main__":
    main()
