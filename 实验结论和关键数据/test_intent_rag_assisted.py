import argparse
import asyncio
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag import ManualRAG, RetrievalResult
from rag_vote_route_test import global_retrieve, load_all_indexes
from util import parse_json_object
from model import intent_agent


DEFAULT_IDS = [
    "190", "196", "197", "198",
    "243", "244", "246", "255", "257", "275",
    "280", "281", "282", "284", "286", "287", "290", "291", "292", "293", "294", "295",
    "298",
    "351", "352", "354", "355", "356", "357",
    "413", "414", "415",
    "427", "430", "432", "433",
]

EXPECTED = {
    "190": "可编程温控器",
    "196": "Active Noise Canceling Truly Wireless Earphones",
    "197": "VR头显",
    "198": "VR头显",
    "243": "2021 Boat 210FSH SPORT 210FSH DELUXE",
    "244": "2021 Boat 210FSH SPORT 210FSH DELUXE",
    "246": "2021 Boat 210FSH SPORT 210FSH DELUXE",
    "255": "2021 Boat 210FSH SPORT 210FSH DELUXE",
    "257": "2021 Boat 210FSH SPORT 210FSH DELUXE",
    "275": "2021 Boat 210FSH SPORT 210FSH DELUXE",
    "280": "Digital, single-lens reflex, AF/AE camera",
    "281": "Digital, single-lens reflex, AF/AE camera",
    "282": "Digital, single-lens reflex, AF/AE camera",
    "284": "Digital, single-lens reflex, AF/AE camera",
    "286": "Digital, single-lens reflex, AF/AE camera",
    "287": "Digital, single-lens reflex, AF/AE camera",
    "290": "Digital, single-lens reflex, AF/AE camera",
    "291": "Digital, single-lens reflex, AF/AE camera",
    "292": "Digital, single-lens reflex, AF/AE camera",
    "293": "Digital, single-lens reflex, AF/AE camera",
    "294": "Digital, single-lens reflex, AF/AE camera",
    "295": "Camera Installation Guide",
    "298": "Active Noise Canceling Truly Wireless Earphones",
    "351": "XL490 XL495",
    "352": "XL490 XL495",
    "354": "XL490 XL495",
    "355": "XL490 XL495",
    "356": "XL490 XL495",
    "357": "XL490 XL495",
    "413": "Camera Installation Guide",
    "414": "Camera Installation Guide",
    "415": "Snowmobile",
    "427": "Color Television",
    "430": "Color Television",
    "432": "Color Television",
    "433": "Color Television",
}


def load_questions(path: Path, ids: list[str]) -> list[dict[str, str]]:
    id_set = set(ids)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("id") in id_set:
                rows.append({"id": row["id"], "question": row["question"]})
    rows.sort(key=lambda item: ids.index(item["id"]))
    return rows


def normalize_text(text: str, max_chars: int = 420) -> str:
    text = " ".join(text.replace("<PIC>", " ").split())
    return text[:max_chars]


def aggregate_candidates(
    results: list[tuple[str, RetrievalResult]],
    manual_count: int = 5,
    snippets_per_manual: int = 2,
    score_top_n: int = 3,
) -> list[dict[str, Any]]:
    by_label: dict[str, list[RetrievalResult]] = defaultdict(list)
    for label, result in results:
        by_label[label].append(result)

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
                    "text": normalize_text(result.chunk.text),
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


def build_prompt(question: str, candidates: list[dict[str, Any]]) -> str:
    parts = [
        f"用户问题：{question}",
        "",
        "下面是全局 RAG 从所有手册中召回并聚合后的候选产品手册。",
        "每个候选手册提供 2 个最相关文本片段，不包含图片。",
        "请只根据用户问题、候选手册名称、候选片段判断该问题应交给哪个产品手册处理。",
        "如果候选片段明显不相关，才返回 customer。",
        "输出必须是 JSON：{\"agent_type\":\"expert或customer\",\"product_name\":\"候选手册名或null\",\"reason\":\"简短原因\"}",
        "",
        "候选手册：",
    ]
    for index, candidate in enumerate(candidates, start=1):
        parts.append(
            f"{index}. {candidate['label']} "
            f"(avg={candidate['avg_score']:.4f}, max={candidate['max_score']:.4f})"
        )
        for snippet_index, snippet in enumerate(candidate["snippets"], start=1):
            parts.append(
                f"   片段{snippet_index} score={snippet['score']:.4f}: {snippet['text']}"
            )
    return "\n".join(parts)


def normalize_product_name(product_name: str, labels: list[str]) -> str:
    name = str(product_name or "").strip()
    if not name or name.lower() == "null":
        return ""
    for label in labels:
        if name == label:
            return label
    for label in sorted(labels, key=len, reverse=True):
        if label and label in name:
            return label
    return name


async def classify(question: str, candidates: list[dict[str, Any]]) -> dict[str, str]:
    system_prompt = (
        "你是产品手册路由器。你的任务是根据用户问题和 RAG 候选片段，"
        "判断问题应该交给哪个产品手册专家。"
        "优先选择能直接回答问题的候选手册；不要因为问题是英文就返回 customer；"
        "只有售后、物流、发票、退换货、投诉等非手册问题，或候选证据明显无关时，才返回 customer。"
    )
    last_error = ""
    content = ""
    for attempt in range(1, 4):
        try:
            completion = await intent_agent._client().chat.completions.create(
                model=intent_agent.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_prompt(question, candidates)},
                ],
            )
            content = completion.choices[0].message.content or ""
            if content.strip():
                break
            last_error = "empty response"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                await asyncio.sleep(attempt)
    if not content.strip():
        return {
            "agent_type": "call_error",
            "product_name": "",
            "reason": last_error,
            "raw": last_error,
        }
    try:
        data = parse_json_object(content)
    except Exception:
        data = {"agent_type": "parse_error", "product_name": "", "reason": content[:300]}
    return {
        "agent_type": str(data.get("agent_type") or ""),
        "product_name": str(data.get("product_name") or ""),
        "reason": str(data.get("reason") or ""),
        "raw": content,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("question_public.csv"))
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--output", type=Path, default=Path("intent_36_rag_assisted_top10.csv"))
    parser.add_argument("--manual-count", type=int, default=10)
    parser.add_argument("--snippets-per-manual", type=int, default=2)
    parser.add_argument("--global-top-k", type=int, default=80)
    args = parser.parse_args()

    questions = load_questions(args.questions, args.ids)
    rag = ManualRAG()
    indexes = load_all_indexes(rag)

    fieldnames = [
        "id",
        "question",
        "expected",
        "agent_type",
        "product_name",
        "normalized_product_name",
        "correct",
        "candidate_labels",
        "expected_in_candidates",
        "reason",
        "raw",
    ]
    rows = []
    for index, row in enumerate(questions, start=1):
        results = global_retrieve(rag, indexes, row["question"], args.global_top_k)
        candidates = aggregate_candidates(
            results,
            manual_count=args.manual_count,
            snippets_per_manual=args.snippets_per_manual,
        )
        predicted = await classify(row["question"], candidates)
        expected = EXPECTED.get(row["id"], "")
        candidate_labels = [candidate["label"] for candidate in candidates]
        normalized_product_name = normalize_product_name(predicted["product_name"], candidate_labels)
        correct = predicted["agent_type"] == "expert" and normalized_product_name == expected
        result_row = {
            "id": row["id"],
            "question": row["question"],
            "expected": expected,
            "agent_type": predicted["agent_type"],
            "product_name": predicted["product_name"],
            "normalized_product_name": normalized_product_name,
            "correct": str(correct),
            "candidate_labels": " | ".join(candidate_labels),
            "expected_in_candidates": str(expected in candidate_labels),
            "reason": predicted["reason"],
            "raw": predicted["raw"],
        }
        rows.append(result_row)
        print(
            f"[{index}/{len(questions)}] id={row['id']} "
            f"expected={expected} predicted={predicted['agent_type']}:{normalized_product_name or predicted['product_name']} "
            f"correct={correct}",
            flush=True,
        )

    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    correct_count = sum(row["correct"] == "True" for row in rows)
    expert_count = sum(row["agent_type"] == "expert" for row in rows)
    candidate_hit = sum(row["expected_in_candidates"] == "True" for row in rows)
    print(f"saved to {args.output}")
    print(f"correct={correct_count}/{len(rows)} expert={expert_count}/{len(rows)} candidate_hit={candidate_hit}/{len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
