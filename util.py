import base64
import json
import mimetypes
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal
import ast

import yaml

from rag import RetrievalResult


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
MANUAL_ALIAS_PATH = BASE_DIR / "manual_aliases.yaml"
MANUAL_SUMMARY_PATH = BASE_DIR / "manual_summaries.yaml"
MANUAL_DIR = BASE_DIR / "手册"
IMAGE_DIR = MANUAL_DIR / "插图"
INTENT_LOG_PATH = BASE_DIR / "intent_log.jsonl"
EXPERT_LOG_PATH = BASE_DIR / "expert_log.jsonl"
MAX_EXPERT_IMAGES = 100

AgentType = Literal["customer", "expert"]
ChatMessage = Dict[str, str]


@dataclass
class AgentConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    prompt_template: str = ""


@dataclass
class IntentResult:
    agent_type: AgentType
    product_name: str | None = None
    manual_path: Path | None = None
    manual_content: str = ""
    image_names: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ManualContent:
    content: str
    image_names: List[str]


@dataclass
class ManualMatch:
    product_name: str
    manual_path: Path
    manual_content: str
    image_names: List[str]


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_manual_aliases(alias_path: Path = MANUAL_ALIAS_PATH) -> Dict[str, List[str]]:
    if not alias_path.exists():
        return {}

    with alias_path.open("r", encoding="utf-8") as file:
        aliases = yaml.safe_load(file) or {}

    normalized_aliases: Dict[str, List[str]] = {}
    for product_name, alias_value in aliases.items():
        if isinstance(alias_value, list):
            normalized_aliases[str(product_name)] = [
                str(alias)
                for alias in alias_value
                if str(alias).strip()
            ]
            continue

        if isinstance(alias_value, dict):
            flattened_aliases: List[str] = []
            for sub_product_name, sub_aliases in alias_value.items():
                flattened_aliases.append(str(sub_product_name))
                if isinstance(sub_aliases, list):
                    flattened_aliases.extend(
                        str(alias)
                        for alias in sub_aliases
                        if str(alias).strip()
                    )
            normalized_aliases[str(product_name)] = flattened_aliases

    return normalized_aliases


def load_manual_sub_aliases(
    alias_path: Path = MANUAL_ALIAS_PATH,
) -> Dict[str, Dict[str, List[str]]]:
    if not alias_path.exists():
        return {}

    with alias_path.open("r", encoding="utf-8") as file:
        aliases = yaml.safe_load(file) or {}

    sub_aliases: Dict[str, Dict[str, List[str]]] = {}
    for product_name, alias_value in aliases.items():
        if not isinstance(alias_value, dict):
            continue

        sub_aliases[str(product_name)] = {
            str(sub_product_name): [
                str(alias)
                for alias in sub_product_aliases
                if str(alias).strip()
            ]
            for sub_product_name, sub_product_aliases in alias_value.items()
            if isinstance(sub_product_aliases, list)
        }

    return sub_aliases


def load_manual_summaries(summary_path: Path = MANUAL_SUMMARY_PATH) -> Dict[str, str]:
    if not summary_path.exists():
        return {}

    with summary_path.open("r", encoding="utf-8") as file:
        summaries = yaml.safe_load(file) or {}

    return {
        str(product_name): str(summary).strip()
        for product_name, summary in summaries.items()
        if str(summary).strip()
    }


def get_agent_config(config: Dict[str, Any], agent_name: str) -> AgentConfig:
    common_config = config.get("common", {})
    agent_config = config.get("agents", {}).get(agent_name, {})
    merged_config = {**common_config, **agent_config}

    return AgentConfig(
        api_key=merged_config.get("api_key", ""),
        base_url=merged_config.get("base_url", ""),
        model=merged_config.get("model", ""),
        prompt_template=merged_config.get("prompt_template", ""),
    )


def latest_user_message(history: List[ChatMessage]) -> str:
    for message in reversed(history):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def normalize_match_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)


def record_intent(
    history: List[ChatMessage],
    intent: IntentResult,
    user_id: str | None = None,
) -> None:
    log_item = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "question": latest_user_message(history),
        "agent_type": intent.agent_type,
        "product_name": intent.product_name,
        "manual_path": str(intent.manual_path) if intent.manual_path else None,
        "image_names": intent.image_names,
        "manual_content_length": len(intent.manual_content),
        "reason": intent.reason,
    }

    with INTENT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def record_expert_response(
    *,
    history: List[ChatMessage],
    intent: IntentResult,
    model: str,
    retrieval_results: List[RetrievalResult],
    candidate_image_names: List[str],
    raw_content: str,
    parsed_answer: str,
    parsed_image_names: List[str],
    final_answer: str,
    final_image_names: List[str],
    user_id: str | None = None,
) -> None:
    log_item = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "question": latest_user_message(history),
        "product_name": intent.product_name,
        "manual_path": str(intent.manual_path) if intent.manual_path else None,
        "model": model,
        "retrieval": [
            {
                "rank": index,
                "score": result.score,
                "entry_index": result.chunk.entry_index,
                "start": result.chunk.start,
                "end": result.chunk.end,
                "image_names": result.chunk.image_names,
                "text_preview": normalize_manual_chunk_text(result.chunk.text)[:300],
            }
            for index, result in enumerate(retrieval_results, start=1)
        ],
        "candidate_image_names": candidate_image_names,
        "raw_content": raw_content,
        "parsed_answer": parsed_answer,
        "parsed_image_names": parsed_image_names,
        "final_answer": final_answer,
        "final_image_names": final_image_names,
    }

    with EXPERT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def parse_json_object(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def load_manual_entries(manual_path: Path) -> List[ManualContent]:
    try:
        raw_text = manual_path.read_text(encoding="utf-8")
    except OSError:
        return []

    parsed_items: List[Any] = []
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

    entries: List[ManualContent] = []
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
            ManualContent(
                content=str(item[0]),
                image_names=image_names,
            )
        )

    return entries


def build_expert_prompt(
    history: List[ChatMessage],
    intent: IntentResult,
    results: List[RetrievalResult],
    image_names: List[str],
) -> str:
    image_labels = build_image_label_map(image_names)
    context_blocks = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        context_blocks.append(
            "\n".join(
                [
                    f"[片段{index}] score={result.score:.4f} "
                    f"entry={chunk.entry_index} span={chunk.start}:{chunk.end}",
                    f"图片标签：{format_image_labels(chunk.image_names, image_labels)}",
                    annotate_pic_tokens(chunk.text, chunk.image_names, image_labels),
                ]
            )
        )

    return (
        f"产品名称：{intent.product_name or '相关产品'}\n"
        f"用户问题：{latest_user_message(history)}\n\n"
        "检索到的手册片段如下。片段中的 <PIC_序号: 图片名> 表示该位置有对应插图。\n"
        "请严格根据手册片段回答，不要编造手册中没有的信息，直接回答问题，回复必须简洁明了。\n"
        "如果答案需要引用图片，只能从“已提供图片”中选择。答案中 <PIC> 数量应等于图片数量。\n"
        "最终 JSON 的 image_names 必须只填写原始图片名，不要填写 PIC 标签。\n"
        "最终只输出 JSON，不要输出 Markdown 或解释。\n"
        'JSON 格式：{"answer":"回答正文，保留必要的 <PIC> 占位符",'
        '"image_names":["图片名","图片名"]}\n\n'
        "参考范例：\n"
        "用户问题：我的DCB107或DCB112型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？\n"
        '输出：{"answer":"DCB107、DCB112 指示灯闪烁标识包括：电池组充电中 <PIC>、电池组已充满 <PIC>、过热/过冷延迟 <PIC>。",'
        '"image_names":["drill0_04","drill0_05","drill0_06"]}\n'
        "用户问题：我想更换健身追踪器的表带，有其他尺寸可选吗？\n"
        '输出：{"answer":"表带有不同尺寸可选，具体尺寸请参考表带尺寸说明。单独销售的配件表带可能略有差异。<PIC>",'
        '"image_names":["Manual16_51"]}\n\n'
        f"已提供图片：\n{format_provided_images(image_names, image_labels)}\n\n"
        "手册片段：\n"
        + "\n\n".join(context_blocks)
    )


def unique_image_names(results: List[RetrievalResult]) -> List[str]:
    if not results:
        return []

    image_names: List[str] = []
    seen = set()
    for result in results:
        for image_name, _ in image_positions_in_chunk(result.chunk):
            if image_name in seen:
                continue
            seen.add(image_name)
            image_names.append(image_name)
            if len(image_names) >= MAX_EXPERT_IMAGES:
                return image_names
    return image_names[:MAX_EXPERT_IMAGES]


def build_extractive_expert_answer(results: List[RetrievalResult]) -> str:
    ranked_results = sorted(results, key=lambda item: item.score, reverse=True)
    selected_texts = []
    total_length = 0
    for result in ranked_results:
        if result.score <= 0 and selected_texts:
            continue
        text = normalize_manual_chunk_text(result.chunk.text)
        if not text:
            continue
        selected_texts.append(text)
        total_length += len(text)
        if total_length >= 700:
            break
    return "\n\n".join(selected_texts)[:900]


def normalize_manual_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def image_positions_in_chunk(chunk: Any) -> List[tuple[str, int]]:
    pic_positions = [
        match.start()
        for match in re.finditer(r"<PIC>", chunk.text)
    ]
    return [
        (image_name, chunk.start + pic_position)
        for image_name, pic_position in zip(chunk.image_names, pic_positions)
    ]


def build_image_label_map(image_names: List[str]) -> Dict[str, str]:
    return {
        image_name: f"PIC_{index}"
        for index, image_name in enumerate(image_names, start=1)
    }


def format_image_labels(image_names: List[str], image_labels: Dict[str, str]) -> str:
    labels = [
        f"<{image_labels[image_name]}: {image_name}>"
        for image_name in image_names
        if image_name in image_labels
    ]
    return ", ".join(labels) if labels else "无"


def format_provided_images(image_names: List[str], image_labels: Dict[str, str]) -> str:
    if not image_names:
        return "无"
    return "\n".join(
        f"<{image_labels[image_name]}: {image_name}>"
        for image_name in image_names
        if image_name in image_labels
    )


def annotate_pic_tokens(
    text: str,
    image_names: List[str],
    image_labels: Dict[str, str],
) -> str:
    if "<PIC>" not in text or not image_names:
        return text

    image_iter = iter(image_names)

    def replace_pic(match: re.Match[str]) -> str:
        image_name = next(image_iter, "")
        label = image_labels.get(image_name)
        if not image_name or not label:
            return match.group(0)
        return f"<{label}: {image_name}>"

    return re.sub(r"<PIC>", replace_pic, text, count=len(image_names))


def parse_expert_json(content: str) -> tuple[str, List[str]]:
    content = content.strip()
    data = parse_json_object(content)
    if not isinstance(data, dict) or not data:
        return content, []

    answer = str(data.get("answer") or "").strip()
    raw_image_names = data.get("image_names") or []
    if not isinstance(raw_image_names, list):
        raw_image_names = []
    return answer, [str(image_name) for image_name in raw_image_names if str(image_name).strip()]


def normalize_answer_pic_tags(answer: str) -> str:
    answer = re.sub(r"<PIC_\d+\s*:\s*[^>]+>", "<PIC>", answer)
    return re.sub(r"<PIC_\d+>", "<PIC>", answer)


def find_image_path(image_name: str) -> Path | None:
    suffixes = ("", ".jpg", ".jpeg", ".png", ".webp", ".bmp")
    image_candidate = Path(image_name)
    for suffix in suffixes:
        candidate = image_candidate if image_candidate.suffix else image_candidate.with_suffix(suffix)
        image_path = candidate if candidate.is_absolute() else IMAGE_DIR / candidate
        if image_path.exists():
            return image_path
    return None


def image_path_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"
