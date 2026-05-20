import argparse
import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def add(blocks: list[tuple[int, int, str]], start: int, end: int, label: str) -> None:
    blocks.append((start, end, label))


def build_expected_blocks() -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    for start, end in [(1, 4), (6, 26), (33, 55), (57, 58)]:
        add(blocks, start, end, "customer")

    add(blocks, 64, 69, "吹风机")
    add(blocks, 70, 85, "空调")
    add(blocks, 86, 88, "蒸汽清洁机")
    add(blocks, 89, 91, "人体工学椅")
    add(blocks, 92, 103, "洗碗机")
    add(blocks, 104, 112, "空气净化器")
    add(blocks, 113, 122, "健身单车")
    add(blocks, 123, 130, "电钻")
    add(blocks, 131, 144, "健身追踪器")
    add(blocks, 145, 146, "冰箱")
    add(blocks, 153, 172, "发电机")
    add(blocks, 173, 180, "摩托艇")
    add(blocks, 181, 185, "水泵")
    add(blocks, 186, 194, "可编程温控器")
    add(blocks, 195, 195, "VR头显")
    add(blocks, 196, 196, "Active Noise Canceling Truly Wireless Earphones")
    add(blocks, 197, 199, "VR头显")
    add(blocks, 200, 205, "功能键盘")
    add(blocks, 206, 207, "儿童电动摩托车")
    add(blocks, 208, 215, "蓝牙激光鼠标")
    add(blocks, 216, 227, "烤箱")
    add(blocks, 228, 234, "相机")
    add(blocks, 241, 241, "Airfryer")
    add(blocks, 242, 264, "2021 Boat 210FSH SPORT 210FSH DELUXE")
    add(blocks, 265, 270, "Espresso")
    add(blocks, 271, 279, "2021 Boat 210FSH SPORT 210FSH DELUXE")
    add(blocks, 280, 294, "Digital, single-lens reflex, AF/AE camera")
    add(blocks, 295, 295, "Camera Installation Guide")
    add(blocks, 296, 302, "Active Noise Canceling Truly Wireless Earphones")
    add(blocks, 303, 310, "E Reader")
    add(blocks, 311, 316, "MFC-J6955DW")
    add(blocks, 317, 321, "Grill")
    add(blocks, 322, 350, "2005 WaveRunner")
    add(blocks, 351, 357, "XL490 XL495")
    add(blocks, 358, 365, "Riding lawn mower")
    add(blocks, 366, 373, "Over-the-Range Microwave")
    add(blocks, 374, 386, "Motherboard")
    add(blocks, 387, 400, "MULTI-USE PRESSURE COOKER AND AIR FRYER-5.7AND8LITRE")
    add(blocks, 401, 412, "VACUUM")
    add(blocks, 413, 414, "Camera Installation Guide")
    add(blocks, 415, 426, "Snowmobile")
    add(blocks, 427, 433, "Color Television")
    add(blocks, 434, 436, "Toothbrush")
    return blocks


def expected_label(qid: int, blocks: list[tuple[int, int, str]]) -> str | None:
    for start, end, label in blocks:
        if start <= qid <= end:
            return label
    return None


def read_questions() -> dict[int, str]:
    questions: dict[int, str] = {}
    with (BASE_DIR / "question_public.csv").open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if row and row[0].isdigit():
                questions[int(row[0])] = row[1].replace("\n", " / ")
    return questions


def read_latest_intents() -> dict[int, str]:
    intents: dict[int, str] = {}
    with (BASE_DIR / "intent_log.jsonl").open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
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
            agent_type = str(item.get("agent_type") or "")
            product_name = str(item.get("product_name") or "")
            intents[qid] = product_name if agent_type == "expert" else "customer"
    return intents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-id", type=int)
    parser.add_argument("--min-id", type=int)
    args = parser.parse_args()

    questions = read_questions()
    intents = read_latest_intents()
    blocks = build_expected_blocks()

    wrong = []
    missing = []
    ok = []
    for qid, question in sorted(questions.items()):
        if args.min_id is not None and qid < args.min_id:
            continue
        if args.max_id is not None and qid > args.max_id:
            continue
        expected = expected_label(qid, blocks)
        actual = intents.get(qid)
        if actual is None:
            missing.append(qid)
        elif expected == actual:
            ok.append(qid)
        else:
            wrong.append((qid, expected, actual, question))

    print(f"questions={len(questions)} intent_logged={len(intents)}")
    print(f"ok={len(ok)} wrong={len(wrong)} missing={len(missing)}")
    if missing:
        print(f"missing={missing}")
    print("\nwrong:")
    for qid, expected, actual, question in wrong:
        print(f"{qid}: expected={expected} actual={actual} question={question[:160]}")


if __name__ == "__main__":
    main()
