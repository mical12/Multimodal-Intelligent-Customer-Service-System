import argparse
import asyncio
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from model import ExpertAgent
from rag import (
    RetrievalResult,
    get_default_rag,
    merge_retrieval_results,
    retrieval_key,
)
from util import IntentResult, get_agent_config, load_config, parse_json_object
from util import find_image_path, image_path_to_data_url


DEFAULT_INPUT = Path("intent_full_qwen37plus.csv")
DEFAULT_OUTPUT = Path("expert_second_retrieval_experiment.csv")


def chunk_id(result: RetrievalResult) -> str:
    chunk = result.chunk
    return f"{chunk.entry_index}:{chunk.start}:{chunk.end}"


def parse_chunk_id(value: str) -> tuple[int, int, int] | None:
    try:
        entry_index, start, end = str(value or "").strip().split(":")
        return int(entry_index), int(start), int(end)
    except ValueError:
        return None


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
    keep_keys = {parse_chunk_id(chunk_id) for chunk_id in keep_chunk_ids}
    keep_keys.discard(None)
    filtered = [
        result
        for result in results
        if retrieval_key(result) in keep_keys
    ]
    return filtered or results


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


def looks_english(text: str) -> bool:
    value = str(text or "")
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in value)
    has_latin = any(char.isascii() and char.isalpha() for char in value)
    return has_latin and not has_cjk


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
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path_to_data_url(image_path)},
            }
        )
    return content


def append_image_name_hints(prompt: str, image_names: list[str]) -> str:
    if not image_names:
        return prompt
    return prompt + "\n\n当前chunk包含图片名，但本轮规划不发送图片内容：\n" + "\n".join(
        f"- {image_name}" for image_name in image_names
    )


def load_expert_rows(
    path: Path,
    limit: int | None,
    sample_size: int | None = None,
    seed: int = 42,
    english_only: bool = False,
) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("agent_type") != "expert":
                continue
            if not row.get("manual_path") or not row.get("product_name"):
                continue
            if english_only and not looks_english(row.get("question", "")):
                continue
            rows.append(row)
    if sample_size:
        rng = random.Random(seed)
        rows = rng.sample(rows, min(sample_size, len(rows)))
        rows.sort(key=lambda row: int(row.get("id") or 0))
    if limit:
        rows = rows[:limit]
    return rows


def make_intent(row: dict[str, str], manual_dir: Path) -> IntentResult:
    manual_name = row["manual_path"]
    manual_path = manual_dir / manual_name
    if not manual_path.exists():
        product_name = row.get("product_name", "")
        for path in manual_dir.glob("*手册.txt"):
            manual_product = path.stem.removesuffix("手册")
            if manual_product == product_name:
                manual_path = path
                break
        else:
            english_manual = manual_dir / "英文汇总手册.txt"
            if english_manual.exists():
                manual_path = english_manual
    return IntentResult(
        agent_type="expert",
        product_name=row["product_name"],
        manual_path=manual_path,
        manual_content="",
        image_names=[],
        reason="second retrieval experiment",
    )


def merge_results(primary: list[RetrievalResult], secondary: list[RetrievalResult]) -> list[RetrievalResult]:
    merged = merge_retrieval_results(primary, secondary)
    merged.sort(key=lambda item: (-item.score, item.chunk.entry_index, item.chunk.start, item.chunk.end))
    return merged


def average_end_distance(chunk: Any, anchor_results: list[RetrievalResult]) -> float:
    anchor_ends = [
        result.chunk.end
        for result in anchor_results
        if result.chunk.entry_index == chunk.entry_index
    ]
    if not anchor_ends:
        return 0.0
    return sum(abs(chunk.end - anchor_end) for anchor_end in anchor_ends) / len(anchor_ends)


async def plan_next_action(
    agent: ExpertAgent,
    *,
    question: str,
    product_name: str,
    results: list[RetrievalResult],
    stage: str,
    step: int,
    action_history: list[str],
    with_images: bool,
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
        "如果问题或片段中出现明确小节名/标题关键词（如启动与停机、热机启动、故障排除、控制设置），"
        "请用search_section输入1到3个标题词、专有名词或核心部件名定位正文。"
        "search_section不要输入完整问题，不要输入泛词，例如steps、procedure、finish、test、machine、how、what；"
        "优先输入能出现在标题或正文中的具体词组，例如Control Set-Up、Unloading、CP Direct Style。"
        "每次调用工具前，请先选择当前片段中哪些chunk_id还值得保留，写入keep_chunk_ids；"
        "只丢弃明显与问题无关、重复、或已确认不需要的chunk。"
        "如果某个chunk可能包含答案、步骤上下文、表格续段或图片说明，必须保留；拿不准也保留。"
        "不要为了压缩上下文而丢弃仍可能有用的chunk。"
        "每轮都必须填写missing_info，说明当前片段距离完整回答还缺什么；"
        "如果coverage=full，missing_info写空字符串。"
        "action_history会包含上一轮工具query和missing_info；不要重复已经new=0的同一个工具动作。"
        "如果search_section返回了接近目标的chunk，不要立刻用相似关键词重复搜索，先expand_neighbors看相邻chunk。"
        "如果expand_neighbors连续没有新增内容，再换search_section或停止。"
        "工具最多会执行6步，所以每一步都要选择最可能补全答案的动作。"
        "只输出JSON，不要输出Markdown。"
    )
    user_prompt = (
        f"stage: {stage}\n"
        f"step: {step}\n"
        f"product_name: {product_name}\n"
        f"question: {question}\n\n"
        f"action_history:\n{chr(10).join(action_history) if action_history else '无'}\n\n"
        f"current_chunks:\n{result_summary(results)}\n\n"
        "available_tools:\n"
        "- expand_neighbors(chunk_id,before_count,after_count): 按手册全文顺序展开指定chunk前后若干chunk；before_count/after_count取0到4，缺开头就加大before_count，缺后续步骤/续表/图片说明就加大after_count。\n"
        "- search_section(query): 输入1到3个具体关键词/标题词/专有名词，不要输入完整问题；在当前手册内用BM25重新召回正文。\n"
        "- none: 当前片段已经足够，停止。\n\n"
        '输出JSON：{"coverage":"full|partial|miss","reason":"简短原因",'
        '"missing_info":"当前chunk还缺的关键信息；如果已经足够则为空字符串",'
        '"keep_chunk_ids":["工具调用前仍有用、需要保留的chunk_id"],'
        '"tool":"none|expand_neighbors|search_section","query":"用于search_section的1到3个关键词，否则为空",'
        '"chunk_id":"用于expand_neighbors的chunk_id，否则为空",'
        '"before_count":2,"after_count":2}'
    )
    image_names = image_names_from_results(results, max_images) if with_images else []
    original_image_count = len(image_names)
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
                image_names = []
                image_fallback = True
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt + "\n\n注意：图片发送失败，本次仅根据文本和图片名判断。",
                    },
                ]
                continue
            if attempt >= 3:
                raise
            await asyncio.sleep(3 * attempt)
    plan = parse_json_object(completion.choices[0].message.content or "")
    plan["_image_count"] = original_image_count
    plan["_image_fallback"] = image_fallback
    return plan


def tool_retrieve(
    agent: ExpertAgent,
    intent: IntentResult,
    query: str,
    top_k: int,
) -> list[RetrievalResult]:
    if not query.strip():
        return []
    return get_default_rag().retrieve_with_bm25_rerank(
        question=query,
        manual_path=intent.manual_path,
        product_name=intent.product_name,
        embedding_top_k=agent.retrieval_candidate_top_k,
        bm25_top_k=agent.retrieval_bm25_top_k,
        final_top_k=top_k,
        neighbor_count=agent.retrieval_neighbor_count,
        reranker_model_dir=agent.reranker_model_dir,
        reranker_max_length=agent.reranker_max_length,
        reranker_batch_size=agent.reranker_batch_size,
    )


def tool_expand_neighbors(
    intent: IntentResult,
    current_results: list[RetrievalResult],
    target_chunk_id: str,
    before_count: int = 2,
    after_count: int = 2,
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
    return [
        RetrievalResult(chunk=chunk, score=base_score)
        for chunk in chunks[start_index : end_index + 1]
    ]


def tool_search_section(
    agent: ExpertAgent,
    intent: IntentResult,
    query: str,
    anchor_results: list[RetrievalResult],
    top_k: int = 4,
) -> list[RetrievalResult]:
    if intent.manual_path is None or not query.strip():
        return []
    return get_default_rag().retrieve_bm25(
        question=query,
        manual_path=intent.manual_path,
        product_name=intent.product_name,
        top_k=top_k,
    )


async def run_one(
    agent: ExpertAgent,
    row: dict[str, str],
    manual_dir: Path,
    second_top_k: int,
    max_steps: int,
    with_images: bool,
    max_images: int,
) -> dict[str, Any]:
    question = row["question"].strip().strip('"')
    intent = make_intent(row, manual_dir)
    first_results = agent._retrieve_manual_context(question, intent)
    current_results = first_results
    added_results: list[RetrievalResult] = []
    actions: list[str] = []
    final_plan: dict[str, Any] = {}
    failed_action_keys: set[tuple[str, str, str]] = set()
    max_sent_image_count = 0
    image_fallback_count = 0

    for step in range(1, max_steps + 1):
        plan = await plan_next_action(
            agent,
            question=question,
            product_name=row["product_name"],
            results=current_results,
            stage="tool_loop",
            step=step,
            action_history=actions,
            with_images=with_images,
            max_images=max_images,
        )
        final_plan = plan
        sent_image_count = int(plan.get("_image_count") or 0)
        max_sent_image_count = max(max_sent_image_count, sent_image_count)
        if plan.get("_image_fallback"):
            image_fallback_count += 1
        coverage = str(plan.get("coverage") or "").lower()
        missing_info = str(plan.get("missing_info") or "").strip()
        tool = str(plan.get("tool") or "none").strip().lower()
        query = str(plan.get("query") or "").strip()
        target_chunk_id = str(plan.get("chunk_id") or "").strip()
        before_count = bounded_int(plan.get("before_count"), default=2, minimum=0, maximum=4)
        after_count = bounded_int(plan.get("after_count"), default=2, minimum=0, maximum=4)
        keep_chunk_ids = normalize_keep_chunk_ids(plan.get("keep_chunk_ids"))
        before_prune_count = len(current_results)
        current_results = filter_results_by_chunk_ids(current_results, keep_chunk_ids)
        kept_count = len(current_results)
        keep_note = f" keep={kept_count}/{before_prune_count}"
        missing_note = f" missing={missing_info}" if missing_info else ""

        if coverage == "full" or tool == "none":
            actions.append(f"{step}:stop coverage={coverage or 'unknown'}{keep_note}{missing_note}")
            break

        action_key = (tool, query, target_chunk_id)
        if action_key in failed_action_keys:
            actions.append(f"{step}:stop repeated_failed_action tool={tool}{keep_note}{missing_note}")
            break

        if tool == "expand_neighbors":
            tool_results = tool_expand_neighbors(
                intent,
                current_results,
                target_chunk_id,
                before_count=before_count,
                after_count=after_count,
            )
            action_detail = (
                f"{step}:expand_neighbors chunk_id={target_chunk_id} "
                f"before={before_count} after={after_count}"
            )
        elif tool == "search_section":
            tool_results = tool_search_section(
                agent,
                intent,
                query,
                first_results[:4],
                top_k=second_top_k,
            )
            action_detail = f"{step}:search_section query={query}"
        elif tool == "retrieve":
            tool_results = tool_search_section(
                agent,
                intent,
                query,
                first_results[:4],
                top_k=second_top_k,
            )
            action_detail = f"{step}:search_section(query_from_retrieve) query={query}"
        else:
            tool_results = []
            action_detail = f"{step}:unknown_tool tool={tool}"

        current_keys = {retrieval_key(result) for result in current_results}
        new_results = [
            result
            for result in tool_results
            if retrieval_key(result) not in current_keys
        ]
        added_results.extend(new_results)
        current_results = merge_results(current_results, new_results)
        actions.append(f"{action_detail} new={len(new_results)} coverage={coverage}{keep_note}{missing_note}")
        if not new_results:
            failed_action_keys.add(action_key)

    return {
        "id": row.get("id", ""),
        "question": question,
        "product_name": row.get("product_name", ""),
        "manual_path": row.get("manual_path", ""),
        "with_images": str(with_images),
        "max_sent_image_count": str(max_sent_image_count),
        "image_fallback_count": str(image_fallback_count),
        "final_coverage": final_plan.get("coverage", ""),
        "final_reason": final_plan.get("reason", ""),
        "final_missing_info": final_plan.get("missing_info", ""),
        "final_tool": final_plan.get("tool", ""),
        "final_query": final_plan.get("query", ""),
        "final_chunk_id": final_plan.get("chunk_id", ""),
        "final_keep_chunk_ids": " | ".join(normalize_keep_chunk_ids(final_plan.get("keep_chunk_ids"))),
        "step_count": str(len(actions)),
        "added_count": str(len(added_results)),
        "final_context_count": str(len(current_results)),
        "actions": " || ".join(actions),
        "first_chunk_ids": " | ".join(chunk_id(result) for result in first_results),
        "added_chunk_ids": " | ".join(chunk_id(result) for result in added_results),
        "merged_chunk_ids": " | ".join(chunk_id(result) for result in current_results),
        "first_preview": result_summary(first_results),
        "added_preview": result_summary(added_results),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--second-top-k", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--english-only", action="store_true")
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--write-batch-size", type=int)
    args = parser.parse_args()

    config = load_config()
    manual_dir = Path(config.get("manual_dir", "手册"))
    agent = ExpertAgent(get_agent_config(config, "expert"))
    rows = load_expert_rows(
        args.input,
        args.limit,
        args.sample_size,
        args.seed,
        english_only=args.english_only,
    )

    fieldnames = [
        "id",
        "question",
        "product_name",
        "manual_path",
        "with_images",
        "max_sent_image_count",
        "image_fallback_count",
        "final_coverage",
        "final_reason",
        "final_missing_info",
        "final_tool",
        "final_query",
        "final_chunk_id",
        "final_keep_chunk_ids",
        "step_count",
        "added_count",
        "final_context_count",
        "actions",
        "first_chunk_ids",
        "added_chunk_ids",
        "merged_chunk_ids",
        "first_preview",
        "added_preview",
    ]

    async def run_indexed(index: int, row: dict[str, str]) -> tuple[int, dict[str, Any]]:
        result = await run_one(
            agent,
            row,
            manual_dir,
            args.second_top_k,
            args.max_steps,
            args.with_images,
            args.max_images,
        )
        return index, result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        completed = 0
        concurrency = max(1, args.concurrency)
        batch_size = max(1, args.write_batch_size or concurrency)
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]
            pending: list[tuple[int, dict[str, str]]] = [
                (batch_start + offset + 1, row)
                for offset, row in enumerate(batch)
            ]
            batch_results: list[tuple[int, dict[str, Any]]] = []
            for group_start in range(0, len(pending), concurrency):
                group = pending[group_start : group_start + concurrency]
                batch_results.extend(
                    await asyncio.gather(
                        *[run_indexed(index, row) for index, row in group]
                    )
                )

            for index, result in sorted(batch_results, key=lambda item: item[0]):
                completed += 1
                writer.writerow(result)
                print(
                    f"[{index}/{len(rows)}] id={result['id']} "
                    f"final={result['final_coverage'] or '-'} "
                    f"steps={result['step_count']} added={result['added_count']} "
                    f"context={result['final_context_count']}"
                )
            file.flush()
            print(f"flushed {completed}/{len(rows)} rows to {args.output}")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
