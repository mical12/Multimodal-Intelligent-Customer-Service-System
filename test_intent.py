import argparse
import asyncio
import csv
from pathlib import Path

from model import intent_agent


DEFAULT_QUESTIONS = [
    "How can you set the jumpers of a motherboard?",
    "What is a V-BeltHolder?",
    "What are the steps to properly connect and set up an outdoor antenna for optimal signal reception?",
    "How to change the default setting of the energy saving mode?",
    "How can you use the washer drain hose?",
    "如何使用蒸汽清洁机安装延长管？",
    "请问你们家的商品支持7天无理由退换货吗？",
]


def load_questions(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return [
            {"id": str(index), "question": question}
            for index, question in enumerate(DEFAULT_QUESTIONS, start=1)
        ]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = []
            for index, row in enumerate(reader, start=1):
                question = row.get("question") or row.get("问题") or ""
                question_id = row.get("id") or row.get("question_id") or str(index)
                if question.strip():
                    rows.append({"id": question_id, "question": question})
            return rows

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            question = line.strip()
            if question:
                rows.append({"id": str(index), "question": question})
    return rows


def write_results(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "id",
        "question",
        "agent_type",
        "product_name",
        "manual_path",
        "manual_content_length",
        "image_count",
        "first_images",
        "reason",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def test_question(row: dict[str, str]) -> dict[str, str]:
    intent = await intent_agent.recognize([
        {"role": "user", "content": row["question"]},
    ])
    return {
        "id": row["id"],
        "question": row["question"],
        "agent_type": intent.agent_type,
        "product_name": intent.product_name or "",
        "manual_path": intent.manual_path.name if intent.manual_path else "",
        "manual_content_length": str(len(intent.manual_content)),
        "image_count": str(len(intent.image_names)),
        "first_images": ", ".join(intent.image_names[:5]),
        "reason": intent.reason,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("intent_test_result.csv"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    questions = load_questions(args.input)
    if args.limit:
        questions = questions[: args.limit]
    results = []
    for index, row in enumerate(questions, start=1):
        result = await test_question(row)
        results.append(result)
        print(
            f"[{index}/{len(questions)}] id={result['id']} "
            f"{result['agent_type']} {result['product_name']} "
            f"{result['manual_path']} images={result['image_count']}"
        )

    write_results(results, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
