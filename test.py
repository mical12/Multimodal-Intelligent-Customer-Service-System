import argparse
import asyncio
import csv
from pathlib import Path

import httpx


API_URL = "http://127.0.0.1:8000/chat"
QUESTION_PATH = Path("question_public.csv")
OUTPUT_PATH = Path("submission.csv")
DEFAULT_BATCH_SIZE = 3
FALLBACK_REPLY = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"


def load_questions(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


async def request_reply(
    client: httpx.AsyncClient,
    question_id: str,
    question: str,
) -> dict[str, str]:
    try:
        response = await client.post(
            API_URL,
            json={
                "user_id": f"test_{question_id}",
                "message": question,
            },
        )
        response.raise_for_status()
        reply = response.json()["reply"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        print(f"id={question_id} failed: {exc}")
        reply = FALLBACK_REPLY

    return {
        "id": question_id,
        "ret": reply,
    }


async def generate_submission_rows(questions, batch_size: int):
    submission_rows = []
    timeout = httpx.Timeout(120.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            end = start + len(batch)
            print(f"[{start + 1}-{end}/{len(questions)}] generating batch")

            tasks = [
                request_reply(client, row["id"], row["question"])
                for row in batch
            ]
            submission_rows.extend(await asyncio.gather(*tasks))

    return submission_rows


def write_submission(rows, path: Path):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


async def main():
    args = parse_args()
    questions = load_questions(QUESTION_PATH)
    submission_rows = await generate_submission_rows(questions, args.batch_size)
    write_submission(submission_rows, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
