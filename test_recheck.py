import argparse
import asyncio
import csv
import time
from pathlib import Path

import httpx


DEFAULT_IDS = ["67", "90", "98", "105", "109", "114", "131", "132", "228", "293"]
QUESTION_PATH = Path("question_public.csv")
OUTPUT_PATH = Path("retest_result.csv")
API_URL = "http://127.0.0.1:8000/chat"


def load_questions(path: Path, ids: list[str]) -> list[dict[str, str]]:
    id_set = set(ids)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row.get("id") in id_set
        ]
    rows.sort(key=lambda row: ids.index(row["id"]))
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "id",
        "question",
        "mode",
        "status",
        "elapsed",
        "agent_type",
        "product_name",
        "manual_name",
        "ret",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def run_direct(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    from model import generate_reply, intent_agent

    results = []
    for row in questions:
        question_id = row["id"]
        question = row["question"]
        history = [{"role": "user", "content": question}]
        started = time.perf_counter()
        try:
            intent = await intent_agent.recognize(history)
            reply = await generate_reply(history, user_id=f"retest_{question_id}")
            elapsed = time.perf_counter() - started
            result = {
                "id": question_id,
                "question": question,
                "mode": "direct",
                "status": "ok",
                "elapsed": f"{elapsed:.2f}",
                "agent_type": intent.agent_type,
                "product_name": intent.product_name or "",
                "manual_name": intent.manual_path.name if intent.manual_path else "",
                "ret": reply,
                "error": "",
            }
        except Exception as exc:
            elapsed = time.perf_counter() - started
            result = {
                "id": question_id,
                "question": question,
                "mode": "direct",
                "status": "error",
                "elapsed": f"{elapsed:.2f}",
                "agent_type": "",
                "product_name": "",
                "manual_name": "",
                "ret": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        print_result(result)
        results.append(result)
    return results


async def run_http(
    questions: list[dict[str, str]],
    api_url: str,
    timeout_seconds: float,
) -> list[dict[str, str]]:
    results = []
    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        try:
            response = await client.get(api_url.rsplit("/", 1)[0])
            print(f"server probe: status={response.status_code}")
        except Exception as exc:
            print(f"server probe failed: {type(exc).__name__}: {exc}")

        for row in questions:
            question_id = row["id"]
            question = row["question"]
            started = time.perf_counter()
            try:
                response = await client.post(
                    api_url,
                    json={
                        "user_id": f"retest_{question_id}",
                        "message": question,
                    },
                )
                response.raise_for_status()
                reply = response.json()["reply"]
                elapsed = time.perf_counter() - started
                result = {
                    "id": question_id,
                    "question": question,
                    "mode": "http",
                    "status": "ok",
                    "elapsed": f"{elapsed:.2f}",
                    "agent_type": "",
                    "product_name": "",
                    "manual_name": "",
                    "ret": reply,
                    "error": "",
                }
            except Exception as exc:
                elapsed = time.perf_counter() - started
                result = {
                    "id": question_id,
                    "question": question,
                    "mode": "http",
                    "status": "error",
                    "elapsed": f"{elapsed:.2f}",
                    "agent_type": "",
                    "product_name": "",
                    "manual_name": "",
                    "ret": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            print_result(result)
            results.append(result)
    return results


def print_result(row: dict[str, str]) -> None:
    print("=" * 80)
    print(f"id={row['id']} mode={row['mode']} status={row['status']} elapsed={row['elapsed']}s")
    print(f"question={row['question']}")
    if row["agent_type"]:
        print(
            "intent="
            f"{row['agent_type']} | {row['product_name']} | {row['manual_name']}"
        )
    if row["error"]:
        print(f"error={row['error']}")
    else:
        print(f"ret={row['ret'][:800].replace(chr(10), ' ')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "http"], default="http")
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--questions", type=Path, default=QUESTION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


async def main():
    args = parse_args()
    questions = load_questions(args.questions, args.ids)
    if not questions:
        raise RuntimeError("没有找到要测试的问题 id")

    if args.mode == "direct":
        rows = await run_direct(questions)
    else:
        rows = await run_http(questions, args.api_url, args.timeout)

    write_rows(args.output, rows)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
