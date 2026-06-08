import asyncio
import re
from dataclasses import dataclass
from typing import Any

from rag import RetrievalResult, get_default_rag, merge_retrieval_results, retrieval_key
from util import IntentResult, find_image_path, image_path_to_data_url, parse_json_object


@dataclass
class AgenticRAGConfig:
    enabled: bool = False
    max_steps: int = 6
    max_images: int = 12
    search_tool: str = "bm25"
    default_before_count: int = 2
    default_after_count: int = 2
    max_before_count: int = 4
    max_after_count: int = 4

    @classmethod
    def from_config(cls, config: dict[str, Any], agent_name: str = "expert") -> "AgenticRAGConfig":
        raw = config.get("agents", {}).get(agent_name, {}).get("agentic_rag", {})
        expand = raw.get("expand_neighbors", {}) if isinstance(raw, dict) else {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            max_steps=int(raw.get("max_steps", 6)),
            max_images=int(raw.get("max_images", 12)),
            search_tool=str(raw.get("search_tool", "bm25")),
            default_before_count=int(expand.get("default_before_count", 2)),
            default_after_count=int(expand.get("default_after_count", 2)),
            max_before_count=int(expand.get("max_before_count", 4)),
            max_after_count=int(expand.get("max_after_count", 4)),
        )


@dataclass
class AgenticRAGResult:
    coverage: str
    reason: str
    missing_info: str
    evidence_summary: str
    results: list[RetrievalResult]
    added_results: list[RetrievalResult]
    actions: list[str]
    image_fallback_count: int
    max_sent_image_count: int


def chunk_id(result: RetrievalResult) -> str:
    chunk = result.chunk
    return f"{chunk.entry_index}:{chunk.start}:{chunk.end}"


def parse_chunk_id(value: str) -> tuple[int, int, int] | None:
    try:
        entry_index, start, end = str(value or "").strip().split(":")
        return int(entry_index), int(start), int(end)
    except ValueError:
        return None


def bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def compact_text(text: str, max_chars: int = 650) -> str:
    return " ".join(str(text or "").split())[:max_chars]


def search_tokens(text: str) -> list[str]:
    text = str(text or "").lower()
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    ascii_tokens = re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text)
    return cjk_tokens + ascii_tokens


def chunk_heading_text(text: str) -> str:
    headings = []
    for match in re.finditer(r"#\s*([^#\n]{1,80})", str(text or "")):
        headings.append(match.group(1).strip())
    if headings:
        return " ".join(headings)
    return compact_text(text, max_chars=120)


def normalize_keep_chunk_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[,，\s|]+", value.strip())
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    else:
        return []
    return [item for item in raw_items if parse_chunk_id(item) is not None]


def filter_results_by_chunk_ids(
    results: list[RetrievalResult],
    keep_chunk_ids: list[str],
) -> list[RetrievalResult]:
    if not keep_chunk_ids:
        return results
    keep_keys = {parse_chunk_id(item) for item in keep_chunk_ids}
    keep_keys.discard(None)
    filtered = [result for result in results if retrieval_key(result) in keep_keys]
    return filtered or results


def merge_results(
    primary: list[RetrievalResult],
    secondary: list[RetrievalResult],
) -> list[RetrievalResult]:
    merged = merge_retrieval_results(primary, secondary)
    merged.sort(key=lambda item: (-item.score, item.chunk.entry_index, item.chunk.start, item.chunk.end))
    return merged


def result_summary(results: list[RetrievalResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            "\n".join(
                [
                    f"[{index}] chunk_id={chunk_id(result)} "
                    f"entry={chunk.entry_index} span={chunk.start}:{chunk.end}",
                    f"images={','.join(chunk.image_names[:8]) or '无'}",
                    compact_text(chunk.text),
                ]
            )
        )
    return "\n\n".join(blocks)


def image_names_from_results(results: list[RetrievalResult], max_images: int) -> list[str]:
    if max_images <= 0:
        return []
    image_names: list[str] = []
    seen = set()
    for result in results:
        for image_name in result.chunk.image_names:
            if image_name in seen:
                continue
            if find_image_path(image_name) is None:
                continue
            seen.add(image_name)
            image_names.append(image_name)
            if len(image_names) >= max_images:
                return image_names
    return image_names


def user_content_with_images(prompt: str, image_names: list[str]) -> str | list[dict[str, Any]]:
    if not image_names:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_name in image_names:
        image_path = find_image_path(image_name)
        if image_path is None:
            continue
        content.append({"type": "text", "text": f"图片：{image_name}"})
        content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path)}})
    return content


def keyword_search(
    intent: IntentResult,
    query: str,
    top_k: int,
) -> list[RetrievalResult]:
    if intent.manual_path is None or not query.strip():
        return []

    index = get_default_rag().load_or_build_index(
        intent.manual_path,
        product_name=intent.product_name,
    )
    chunks = index.get("chunks", [])
    tokens = search_tokens(query)
    if not tokens:
        return []

    scored: list[tuple[float, int, Any]] = []
    for order, chunk in enumerate(chunks):
        text = str(chunk.text or "")
        text_lower = text.lower()
        heading = chunk_heading_text(text).lower()
        score = 0.0
        for token in tokens:
            token_lower = token.lower()
            text_count = text_lower.count(token_lower)
            heading_count = heading.count(token_lower)
            if heading_count:
                score += 8.0 * heading_count
            if text_count:
                score += 1.0 * text_count
        compact_query = "".join(tokens)
        compact_heading = re.sub(r"\s+", "", heading)
        compact_body = re.sub(r"\s+", "", text_lower[:300])
        if compact_query and compact_query in compact_heading:
            score += 12.0
        elif compact_query and compact_query in compact_body:
            score += 6.0
        if score > 0:
            scored.append((score, order, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [RetrievalResult(chunk=chunk, score=score) for score, _, chunk in scored[:top_k]]


def search_section(
    intent: IntentResult,
    query: str,
    top_k: int,
    search_tool: str = "bm25",
) -> list[RetrievalResult]:
    if intent.manual_path is None or not query.strip():
        return []
    if search_tool == "keyword":
        return keyword_search(intent, query, top_k)
    return get_default_rag().retrieve_bm25(
        question=query,
        manual_path=intent.manual_path,
        product_name=intent.product_name,
        top_k=top_k,
    )


def expand_neighbors(
    intent: IntentResult,
    current_results: list[RetrievalResult],
    target_chunk_id: str,
    before_count: int,
    after_count: int,
) -> list[RetrievalResult]:
    target_key = parse_chunk_id(target_chunk_id)
    if target_key is None or intent.manual_path is None:
        return []

    index = get_default_rag().load_or_build_index(
        intent.manual_path,
        product_name=intent.product_name,
    )
    chunks = sorted(
        index.get("chunks", []),
        key=lambda chunk: (chunk.entry_index, chunk.start, chunk.end),
    )
    base_score = 0.0
    for result in current_results:
        if retrieval_key(result) == target_key:
            base_score = result.score
            break

    target_index = None
    for index, chunk in enumerate(chunks):
        if (chunk.entry_index, chunk.start, chunk.end) == target_key:
            target_index = index
            break
    if target_index is None:
        return []

    start_index = max(0, target_index - before_count)
    end_index = min(len(chunks) - 1, target_index + after_count)
    return [RetrievalResult(chunk=chunk, score=base_score) for chunk in chunks[start_index : end_index + 1]]


async def plan_next_action(
    agent: Any,
    *,
    question: str,
    product_name: str,
    results: list[RetrievalResult],
    step: int,
    action_history: list[str],
    max_images: int,
) -> dict[str, Any]:
    system_prompt = (
        "你是产品手册RAG阅读规划器。请判断当前手册片段是否足以回答用户问题，并决定下一步工具。"
        "coverage只能是full、partial、miss三选一。"
        "如果coverage=full，tool必须为none。"
        "如果上一步search_section已经找到接近目标的小节标题、正文片段、表格或图片说明，但内容不完整，"
        "下一步请优先选择expand_neighbors，并填写最值得展开的chunk_id以及before_count/after_count。"
        "如果缺少手册顺序上的前后文、步骤开头/结尾、表格续段或图片说明，也请优先选择expand_neighbors。"
        "如果当前片段主题明显不对、只命中跳转提示、或还没有找到接近目标的小节，请选择search_section。"
        "search_section输入1到3个标题词、专有名词或核心部件名，不要输入完整问题或泛词。"
        "每次调用工具前，请先选择当前片段中哪些chunk_id还值得保留，写入keep_chunk_ids；拿不准就保留。"
        "每轮都必须填写missing_info，说明当前片段距离完整回答还缺什么；如果coverage=full则为空。"
        "action_history会包含上一轮工具query和missing_info；不要重复已经new=0的同一个工具动作。"
        "如果search_section返回了接近目标的chunk，不要立刻用相似关键词重复搜索，先expand_neighbors看相邻chunk。"
        "只输出JSON，不要输出Markdown。"
    )
    user_prompt = (
        f"step: {step}\n"
        f"product_name: {product_name}\n"
        f"question: {question}\n\n"
        f"action_history:\n{chr(10).join(action_history) if action_history else '无'}\n\n"
        f"current_chunks:\n{result_summary(results)}\n\n"
        "available_tools:\n"
        "- expand_neighbors(chunk_id,before_count,after_count): 按手册全文顺序展开指定chunk前后若干chunk；before_count/after_count取0到配置上限。\n"
        "- search_section(query): 输入1到3个具体关键词/标题词/专有名词，在当前手册内重新召回正文。\n"
        "- none: 当前片段已经足够，停止。\n\n"
        '输出JSON：{"coverage":"full|partial|miss","reason":"简短原因",'
        '"missing_info":"当前chunk还缺的关键信息；如果已经足够则为空字符串",'
        '"keep_chunk_ids":["工具调用前仍有用、需要保留的chunk_id"],'
        '"tool":"none|expand_neighbors|search_section","query":"用于search_section的1到3个关键词，否则为空",'
        '"chunk_id":"用于expand_neighbors的chunk_id，否则为空",'
        '"before_count":2,"after_count":2}'
    )

    image_names = image_names_from_results(results, max_images)
    image_fallback = False
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content_with_images(user_prompt, image_names)},
    ]

    for attempt in range(1, 4):
        try:
            completion = await agent._client().chat.completions.create(
                model=agent.model,
                temperature=0,
                messages=messages,
                **agent._chat_completion_options(),
            )
            break
        except Exception as exc:
            if image_names and agent._is_data_inspection_error(exc):
                image_fallback = True
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt + "\n\n注意：图片发送失败，本次仅根据文本和图片名判断。",
                    },
                ]
                image_names = []
                continue
            if attempt >= 3:
                raise
            await asyncio.sleep(3 * attempt)

    plan = parse_json_object(completion.choices[0].message.content or "")
    plan["_image_count"] = len(image_names)
    plan["_image_fallback"] = image_fallback
    return plan


async def summarize_evidence(
    agent: Any,
    *,
    question: str,
    product_name: str,
    results: list[RetrievalResult],
    coverage: str,
    reason: str,
    missing_info: str,
    max_images: int,
) -> tuple[str, bool, int]:
    system_prompt = (
        "You are a product-manual evidence summarizer. "
        "You only summarize the provided chunks; do not answer as the final assistant. "
        "Identify which chunks are useful for answering the user question, what each useful chunk supports, "
        "which images should be referenced if needed, and whether any information is still missing. "
        "Do not copy long source text. Output concise JSON only."
    )
    user_prompt = (
        f"product_name: {product_name}\n"
        f"question: {question}\n"
        f"planner_coverage: {coverage}\n"
        f"planner_reason: {reason}\n"
        f"planner_missing_info: {missing_info}\n\n"
        f"chunks:\n{result_summary(results)}\n\n"
        "Output JSON with this schema:\n"
        '{"useful_chunks":[{"chunk_id":"...","why_useful":"...","key_points":["..."],'
        '"image_names":["..."]}],"irrelevant_chunk_ids":["..."],'
        '"answer_outline":["..."],"image_guidance":"...",'
        '"remaining_gap":"empty if enough, otherwise what is missing"}'
    )

    image_names = image_names_from_results(results, max_images)
    image_fallback = False
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content_with_images(user_prompt, image_names)},
    ]

    for attempt in range(1, 4):
        try:
            completion = await agent._client().chat.completions.create(
                model=agent.model,
                temperature=0,
                messages=messages,
                **agent._chat_completion_options(),
            )
            raw_content = completion.choices[0].message.content or ""
            parsed = parse_json_object(raw_content)
            if parsed:
                return evidence_summary_text(parsed), image_fallback, len(image_names)
            return raw_content.strip(), image_fallback, len(image_names)
        except Exception as exc:
            if image_names and agent._is_data_inspection_error(exc):
                image_fallback = True
                image_names = []
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt + "\n\nImages could not be sent; summarize by text and image names only.",
                    },
                ]
                continue
            if attempt >= 3:
                return "", image_fallback, len(image_names)
            await asyncio.sleep(3 * attempt)

    return "", image_fallback, len(image_names)


def evidence_summary_text(parsed: dict[str, Any]) -> str:
    lines = ["Agentic RAG evidence summary:"]
    useful_chunks = parsed.get("useful_chunks") or []
    if useful_chunks:
        lines.append("Useful chunks:")
        for item in useful_chunks:
            if not isinstance(item, dict):
                continue
            key_points = item.get("key_points") or []
            images = item.get("image_names") or []
            lines.append(
                "- chunk_id={chunk_id}; why={why}; key_points={points}; images={images}".format(
                    chunk_id=item.get("chunk_id", ""),
                    why=item.get("why_useful", ""),
                    points=" | ".join(str(point) for point in key_points),
                    images=", ".join(str(image) for image in images),
                )
            )
    answer_outline = parsed.get("answer_outline") or []
    if answer_outline:
        lines.append("Answer outline:")
        for point in answer_outline:
            lines.append(f"- {point}")
    image_guidance = str(parsed.get("image_guidance") or "").strip()
    if image_guidance:
        lines.append(f"Image guidance: {image_guidance}")
    remaining_gap = str(parsed.get("remaining_gap") or "").strip()
    if remaining_gap:
        lines.append(f"Remaining gap: {remaining_gap}")
    irrelevant = parsed.get("irrelevant_chunk_ids") or []
    if irrelevant:
        lines.append("Irrelevant chunk ids: " + ", ".join(str(item) for item in irrelevant))
    return "\n".join(lines)


async def run_agentic_rag(
    *,
    agent: Any,
    question: str,
    intent: IntentResult,
    initial_results: list[RetrievalResult],
    top_k: int,
    config: AgenticRAGConfig,
) -> AgenticRAGResult:
    current_results = initial_results
    added_results: list[RetrievalResult] = []
    actions: list[str] = []
    final_plan: dict[str, Any] = {}
    evidence_summary = ""
    failed_action_keys: set[tuple[str, str, str]] = set()
    max_sent_image_count = 0
    image_fallback_count = 0

    for step in range(1, config.max_steps + 1):
        plan = await plan_next_action(
            agent,
            question=question,
            product_name=intent.product_name,
            results=current_results,
            step=step,
            action_history=actions,
            max_images=config.max_images,
        )
        final_plan = plan
        max_sent_image_count = max(max_sent_image_count, int(plan.get("_image_count") or 0))
        if plan.get("_image_fallback"):
            image_fallback_count += 1

        coverage = str(plan.get("coverage") or "").lower()
        missing_info = str(plan.get("missing_info") or "").strip()
        tool = str(plan.get("tool") or "none").strip().lower()
        query = str(plan.get("query") or "").strip()
        target_chunk_id = str(plan.get("chunk_id") or "").strip()
        before_count = bounded_int(
            plan.get("before_count"),
            default=config.default_before_count,
            minimum=0,
            maximum=config.max_before_count,
        )
        after_count = bounded_int(
            plan.get("after_count"),
            default=config.default_after_count,
            minimum=0,
            maximum=config.max_after_count,
        )

        before_prune_count = len(current_results)
        current_results = filter_results_by_chunk_ids(
            current_results,
            normalize_keep_chunk_ids(plan.get("keep_chunk_ids")),
        )
        keep_note = f" keep={len(current_results)}/{before_prune_count}"
        missing_note = f" missing={missing_info}" if missing_info else ""

        if coverage == "full" or tool == "none":
            actions.append(f"{step}:stop coverage={coverage or 'unknown'}{keep_note}{missing_note}")
            evidence_summary, evidence_image_fallback, evidence_image_count = await summarize_evidence(
                agent,
                question=question,
                product_name=intent.product_name or "",
                results=current_results,
                coverage=coverage,
                reason=str(plan.get("reason") or ""),
                missing_info=missing_info,
                max_images=config.max_images,
            )
            max_sent_image_count = max(max_sent_image_count, evidence_image_count)
            if evidence_image_fallback:
                image_fallback_count += 1
            break

        action_key = (tool, query, target_chunk_id)
        if action_key in failed_action_keys:
            actions.append(f"{step}:stop repeated_failed_action tool={tool}{keep_note}{missing_note}")
            break

        if tool == "expand_neighbors":
            tool_results = expand_neighbors(
                intent,
                current_results,
                target_chunk_id,
                before_count=before_count,
                after_count=after_count,
            )
            action_detail = f"{step}:expand_neighbors chunk_id={target_chunk_id} before={before_count} after={after_count}"
        elif tool == "search_section":
            tool_results = search_section(
                intent,
                query,
                top_k=top_k,
                search_tool=config.search_tool,
            )
            action_detail = f"{step}:search_section query={query}"
        else:
            tool_results = []
            action_detail = f"{step}:unknown_tool tool={tool}"

        current_keys = {retrieval_key(result) for result in current_results}
        new_results = [result for result in tool_results if retrieval_key(result) not in current_keys]
        added_results.extend(new_results)
        current_results = merge_results(current_results, new_results)
        actions.append(f"{action_detail} new={len(new_results)} coverage={coverage}{keep_note}{missing_note}")
        if not new_results:
            failed_action_keys.add(action_key)

    return AgenticRAGResult(
        coverage=str(final_plan.get("coverage", "")),
        reason=str(final_plan.get("reason", "")),
        missing_info=str(final_plan.get("missing_info", "")),
        evidence_summary=evidence_summary,
        results=current_results,
        added_results=added_results,
        actions=actions,
        image_fallback_count=image_fallback_count,
        max_sent_image_count=max_sent_image_count,
    )
