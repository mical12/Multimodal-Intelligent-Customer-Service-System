import argparse
import asyncio
import csv
from datetime import datetime
from pathlib import Path
import time

import httpx


API_URL = "http://127.0.0.1:8000/chat"
QUESTION_PATH = Path("question_public.csv")
OUTPUT_PATH = Path("submission.csv")
ERROR_LOG_PATH = Path("test_errors.csv")
DEFAULT_BATCH_SIZE = 3
FALLBACK_REPLY = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"


def append_error_log(
    question_id: str,
    question: str,
    error_type: str,
    error_message: str,
    elapsed: float,
    status_code: int | None = None,
    response_text: str = "",
    path: Path = ERROR_LOG_PATH,
) -> None:
    is_new_file = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time",
                "id",
                "question",
                "error_type",
                "error_message",
                "elapsed",
                "status_code",
                "response_text",
            ],
        )
        if is_new_file:
            writer.writeheader()
        writer.writerow(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "id": question_id,
                "question": question,
                "error_type": error_type,
                "error_message": error_message,
                "elapsed": f"{elapsed:.2f}",
                "status_code": status_code or "",
                "response_text": response_text[:1000],
            }
        )


def load_questions(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


async def request_reply(
    client: httpx.AsyncClient,
    question_id: str,
    question: str,
) -> dict[str, str]:
    started = time.perf_counter()
    response: httpx.Response | None = None
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
    except Exception as exc:
        elapsed = time.perf_counter() - started
        status_code = response.status_code if response is not None else None
        response_text = response.text if response is not None else ""
        append_error_log(
            question_id=question_id,
            question=question,
            error_type=type(exc).__name__,
            error_message=str(exc),
            elapsed=elapsed,
            status_code=status_code,
            response_text=response_text,
        )
        print(
            f"id={question_id} failed: {type(exc).__name__}: {exc} "
            f"elapsed={elapsed:.2f}s status={status_code}"
        )
        reply = FALLBACK_REPLY

    return {
        "id": question_id,
        "ret": reply,
    }


async def generate_submission_rows(
    questions,
    batch_size: int,
):
    submission_rows = []
    timeout = httpx.Timeout(120.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
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


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return {row["id"] for row in reader if row.get("id")}


def prepare_output(path: Path, resume: bool):
    if resume and path.exists():
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "ret"])
        writer.writeheader()
    if ERROR_LOG_PATH.exists():
        ERROR_LOG_PATH.unlink()


def append_submission(rows, path: Path):
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "ret"])
        writer.writerows(rows)


async def generate_submission_file(
    questions,
    batch_size: int,
    output_path: Path,
    resume: bool,
):
    done_ids = existing_ids(output_path) if resume else set()
    pending_questions = [
        row
        for row in questions
        if row["id"] not in done_ids
    ]
    prepare_output(output_path, resume)

    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        for start in range(0, len(pending_questions), batch_size):
            batch = pending_questions[start:start + batch_size]
            end = start + len(batch)
            print(f"[{start + 1}-{end}/{len(pending_questions)}] generating batch")

            tasks = [
                request_reply(client, row["id"], row["question"])
                for row in batch
            ]
            rows = await asyncio.gather(*tasks)
            append_submission(rows, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


async def main():
    args = parse_args()
    questions = load_questions(QUESTION_PATH)
    await generate_submission_file(
        questions,
        args.batch_size,
        args.output,
        args.resume,
    )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
