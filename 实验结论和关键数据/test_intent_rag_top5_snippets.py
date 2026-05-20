import argparse
import asyncio
import csv
import re
from pathlib import Path
from typing import Any

from model import intent_agent
from rag import ManualRAG, RetrievalResult
from rag_vote_route_test import global_retrieve, load_all_indexes
from test_intent_rag_assisted import DEFAULT_IDS, EXPECTED, load_questions
from util import parse_json_object


DEFAULT_OUTPUT = Path("intent_36_rag_top5_2snippets.csv")


def pure_text(text: str, max_chars: int = 650) -> str:
    text = text.replace("<PIC>", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def select_evidence(
    top5: list[tuple[str, RetrievalResult]],
    snippet_count: int = 2,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for label, result in top5:
        text = pure_text(result.chunk.text)
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        evidence.append(
            {
                "label": label,
                "score": result.score,
                "entry_index": result.chunk.entry_index,
                "text": text,
            }
        )
        if len(evidence) >= snippet_count:
            break
    return evidence


def build_prompt(
    question: str,
    top5: list[tuple[str, RetrievalResult]],
    evidence: list[dict[str, Any]],
) -> str:
    top5_lines = [
        f"{rank}. {label} score={result.score:.4f} entry={result.chunk.entry_index}"
        for rank, (label, result) in enumerate(top5, start=1)
    ]
    evidence_lines = []
    for index, item in enumerate(evidence, start=1):
        evidence_lines.append(
            f"片段{index} | 手册={item['label']} | score={item['score']:.4f} | "
            f"entry={item['entry_index']}\n{item['text']}"
        )

    return "\n".join(
        [
            f"用户问题：{question}",
            "",
            "下面是全局 RAG 从所有手册中召回的前五名候选：",
            *top5_lines,
            "",
            "下面是从前五名中选出的 2 个最相关纯文本片段，已去掉图片标记：",
            *evidence_lines,
            "",
            "请根据用户问题、前五名候选和这 2 个文本片段判断路由。",
            "如果片段或候选能直接回答该问题，返回 expert，并且 product_name 必须精确使用前五名候选中的一个手册名。",
            "只有售后、物流、发票、退换货、投诉等非手册问题，或前五名和片段明显无关时，才返回 customer。",
            '只输出 JSON：{"agent_type":"expert或customer","product_name":"候选手册名或null","reason":"简短原因"}',
        ]
    )


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


async def classify(
    question: str,
    top5: list[tuple[str, RetrievalResult]],
    evidence: list[dict[str, Any]],
) -> dict[str, str]:
    system_prompt = (
        "你是产品手册路由器。你的任务是根据用户问题和全局 RAG 证据，判断问题应交给哪个产品手册专家。"
        "不要因为问题没有显式产品名就返回 customer；只要证据片段能够回答问题，就选择对应产品手册。"
        "product_name 必须精确匹配候选手册名，不要附加分数、括号或解释。"
    )
    content = ""
    last_error = ""
    for attempt in range(1, 4):
        try:
            completion = await intent_agent._client().chat.completions.create(
                model=intent_agent.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_prompt(question, top5, evidence)},
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
        data = {
            "agent_type": "parse_error",
            "product_name": "",
            "reason": content[:300],
        }

    return {
        "agent_type": str(data.get("agent_type") or "").strip(),
        "product_name": str(data.get("product_name") or "").strip(),
        "reason": str(data.get("reason") or "").strip(),
        "raw": content,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("question_public.csv"))
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--snippet-count", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    questions = load_questions(args.questions, [str(item) for item in args.ids])
    if args.limit:
        questions = questions[: args.limit]
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
        "top5_labels",
        "evidence_labels",
        "expected_in_top5",
        "reason",
        "raw",
    ]
    rows: list[dict[str, str]] = []

    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(questions, start=1):
            top5 = global_retrieve(rag, indexes, row["question"], args.top_k)
            evidence = select_evidence(top5, args.snippet_count)
            predicted = await classify(row["question"], top5, evidence)
            labels = [label for label, _ in top5]
            normalized_product = normalize_product_name(predicted["product_name"], labels)
            expected = EXPECTED.get(row["id"], "")
            correct = predicted["agent_type"] == "expert" and normalized_product == expected

            result_row = {
                "id": row["id"],
                "question": row["question"],
                "expected": expected,
                "agent_type": predicted["agent_type"],
                "product_name": predicted["product_name"],
                "normalized_product_name": normalized_product,
                "correct": str(correct),
                "top5_labels": " | ".join(labels),
                "evidence_labels": " | ".join(item["label"] for item in evidence),
                "expected_in_top5": str(expected in labels),
                "reason": predicted["reason"],
                "raw": predicted["raw"],
            }
            rows.append(result_row)
            writer.writerow(result_row)
            file.flush()
            print(
                f"[{index}/{len(questions)}] id={row['id']} expected={expected} "
                f"predicted={predicted['agent_type']}:{normalized_product or predicted['product_name']} "
                f"correct={correct}",
                flush=True,
            )

    total = len(rows)
    correct_count = sum(row["correct"] == "True" for row in rows)
    expert_count = sum(row["agent_type"] == "expert" for row in rows)
    top5_hit_count = sum(row["expected_in_top5"] == "True" for row in rows)
    print(f"saved to {args.output}")
    print(
        f"correct={correct_count}/{total} "
        f"expert={expert_count}/{total} "
        f"expected_in_top5={top5_hit_count}/{total}"
    )


if __name__ == "__main__":
    asyncio.run(main())
