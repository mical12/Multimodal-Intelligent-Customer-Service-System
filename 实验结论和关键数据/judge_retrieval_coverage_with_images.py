import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import IntentAgent
from rag import MANUAL_DIR, ManualChunk, RetrievalResult, get_default_rag
from util import (
    build_image_label_map,
    annotate_pic_tokens,
    find_image_path,
    get_agent_config,
    image_path_to_data_url,
    image_positions_in_chunk,
    load_config,
    parse_json_object,
)


DEFAULT_INPUT_LOG = Path("expert_log.jsonl")
DEFAULT_OUTPUT = Path("retrieval_coverage_judge_with_images.csv")
DEFAULT_RERANKER_DIR = Path("Qwen3-Reranker-0.6B") / "models" / "Qwen3-Reranker-0.6B"
DEFAULT_VL_RERANKER_DIR = Path("Qwen3-VL-Reranker-2B")
DEFAULT_RERANK_INSTRUCTION = (
    "Given a product manual question, retrieve the manual passage that contains "
    "the information needed to answer the user's question."
)


def normalize_rerank_text(text: str, max_chars: int = 1800) -> str:
    return " ".join(text.replace("<PIC>", " ").split())[:max_chars]


def format_rerank_instruction(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*|[\u4e00-\u9fff]")


def retrieval_key(result: RetrievalResult) -> tuple[int, int, int]:
    chunk = result.chunk
    return (chunk.entry_index, chunk.start, chunk.end)


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text.replace("<PIC>", " "))]


def bm25_retrieve(
    question: str,
    manual_path: Path,
    product_name: str,
    top_k: int,
) -> list[RetrievalResult]:
    index = get_default_rag().load_or_build_index(
        manual_path=manual_path,
        product_name=product_name,
    )
    chunks: list[ManualChunk] = index["chunks"]
    if not chunks:
        return []

    tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    query_tokens = tokenize_for_bm25(question)
    if not query_tokens:
        return []

    doc_freq: dict[str, int] = {}
    for tokens in tokenized_chunks:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    avgdl = sum(len(tokens) for tokens in tokenized_chunks) / max(len(tokenized_chunks), 1)
    k1 = 1.5
    b = 0.75
    query_unique = set(query_tokens)
    results: list[RetrievalResult] = []
    total_docs = len(tokenized_chunks)
    for chunk, tokens in zip(chunks, tokenized_chunks, strict=False):
        if not tokens:
            continue
        tf: dict[str, int] = {}
        for token in tokens:
            if token in query_unique:
                tf[token] = tf.get(token, 0) + 1
        if not tf:
            continue
        score = 0.0
        length_norm = k1 * (1 - b + b * len(tokens) / max(avgdl, 1))
        for token, freq in tf.items():
            df = doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            score += idf * (freq * (k1 + 1)) / (freq + length_norm)
        if score > 0:
            results.append(RetrievalResult(chunk=chunk, score=score))

    return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def merge_retrieval_results(
    primary: list[RetrievalResult],
    secondary: list[RetrievalResult],
) -> list[RetrievalResult]:
    merged: dict[tuple[int, int, int], RetrievalResult] = {}
    for result in primary:
        merged[retrieval_key(result)] = result
    for result in secondary:
        key = retrieval_key(result)
        if key not in merged:
            merged[key] = result
    return list(merged.values())


class QwenReranker:
    def __init__(self, model_dir: Path, max_length: int = 1024, batch_size: int = 2) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            padding_side="left",
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.float32,
        ).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on '
            'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def score_pairs(self, pairs: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            scores.extend(self._score_batch(pairs[start : start + self.batch_size]))
        return scores

    @torch.no_grad()
    def _score_batch(self, pairs: list[str]) -> list[float]:
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=budget,
        )
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self.prefix_tokens + input_ids + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        outputs = self.model(**inputs).logits[:, -1, :]
        true_vector = outputs[:, self.token_true_id]
        false_vector = outputs[:, self.token_false_id]
        logits = torch.stack([false_vector, true_vector], dim=1)
        return torch.nn.functional.log_softmax(logits, dim=1)[:, 1].exp().tolist()


def rerank_results(
    reranker: QwenReranker,
    question: str,
    results: list[RetrievalResult],
    top_k: int,
    instruction: str,
) -> tuple[list[RetrievalResult], str]:
    pairs = [
        format_rerank_instruction(instruction, question, normalize_rerank_text(result.chunk.text))
        for result in results
    ]
    if not pairs:
        return [], ""

    scores = reranker.score_pairs(pairs)
    ranked = sorted(
        zip(results, scores, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    detail = "|".join(
        f"emb={result.score:.6f},rerank={score:.6f},span={result.chunk.start}:{result.chunk.end}"
        for result, score in ranked[:top_k]
    )
    return [result for result, _ in ranked[:top_k]], detail


class QwenVLReranker:
    def __init__(
        self,
        model_dir: Path,
        max_images_per_chunk: int = 2,
        max_chars: int = 1800,
    ) -> None:
        script_dir = model_dir / "scripts"
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from qwen3_vl_reranker import Qwen3VLReranker

        self.model = Qwen3VLReranker(
            model_name_or_path=str(model_dir),
            max_length=4096,
            default_instruction=DEFAULT_RERANK_INSTRUCTION,
        )
        original_tokenize = self.model.tokenize

        def tokenize_with_tensor_token_types(*args: Any, **kwargs: Any) -> Any:
            inputs = original_tokenize(*args, **kwargs)
            token_types = inputs.get("mm_token_type_ids")
            if isinstance(token_types, list):
                inputs["mm_token_type_ids"] = torch.tensor(
                    token_types,
                    dtype=torch.long,
                    device=inputs["input_ids"].device,
                )
            return inputs

        self.model.tokenize = tokenize_with_tensor_token_types
        self.max_images_per_chunk = max_images_per_chunk
        self.max_chars = max_chars

    def rerank(
        self,
        question: str,
        results: list[RetrievalResult],
        top_k: int,
        instruction: str,
    ) -> tuple[list[RetrievalResult], str]:
        scored: list[tuple[RetrievalResult, float, str]] = []
        for result in results:
            documents: list[dict[str, str]] = [
                {"text": normalize_rerank_text(result.chunk.text, self.max_chars)}
            ]
            for image_name in result.chunk.image_names[: self.max_images_per_chunk]:
                image_path = find_image_path(image_name)
                if image_path is None:
                    continue
                documents.append(
                    {
                        "text": normalize_rerank_text(result.chunk.text, self.max_chars),
                        "image": str(image_path),
                    }
                )

            scores = self.model.process(
                {
                    "instruction": instruction,
                    "query": {"text": question},
                    "documents": documents,
                }
            )
            best_score = max(float(score) for score in scores) if scores else 0.0
            scored.append(
                (
                    result,
                    best_score,
                    f"emb={result.score:.6f},vl={best_score:.6f},span={result.chunk.start}:{result.chunk.end},variants={len(documents)}",
                )
            )

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        return [result for result, _, _ in ranked[:top_k]], "|".join(item[2] for item in ranked[:top_k])


def resolve_manual_path(raw_path: str) -> Path:
    manual_path = Path(raw_path)
    if manual_path.exists():
        return manual_path

    by_name = MANUAL_DIR / manual_path.name
    if by_name.exists():
        return by_name

    return manual_path


def load_cases_from_expert_log(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = str(record.get("question") or "").strip()
            product_name = str(record.get("product_name") or "").strip()
            manual_path = resolve_manual_path(str(record.get("manual_path") or ""))
            if not question or not product_name or not manual_path:
                continue
            key = (question, product_name, str(manual_path))
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                {
                    "id": record.get("id") or "",
                    "question": question.strip('"'),
                    "product_name": product_name,
                    "manual_path": manual_path,
                }
            )
    return cases


def load_cases_from_csv(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            question = str(row.get("question") or "").strip()
            product_name = str(
                row.get("product_name")
                or row.get("selected_product")
                or ""
            ).strip()
            manual_path = resolve_manual_path(str(row.get("manual_path") or ""))
            if not question:
                continue
            cases.append(
                {
                    "id": row.get("id") or "",
                    "question": question.strip('"'),
                    "product_name": product_name,
                    "manual_path": manual_path,
                }
            )
    return cases


async def reroute_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config = load_config()
    intent_agent = IntentAgent(get_agent_config(config, "intent"))
    rerouted: list[dict[str, Any]] = []
    for case in cases:
        intent = await intent_agent.recognize(
            [{"role": "user", "content": case["question"]}]
        )
        new_case = dict(case)
        new_case["original_product_name"] = case.get("product_name", "")
        new_case["original_manual_path"] = str(case.get("manual_path", ""))
        new_case["reroute_agent_type"] = intent.agent_type
        new_case["reroute_reason"] = intent.reason
        if intent.agent_type == "expert" and intent.manual_path is not None:
            new_case["product_name"] = intent.product_name or ""
            new_case["manual_path"] = intent.manual_path
        rerouted.append(new_case)
    return rerouted


def collect_image_names(results: list[RetrievalResult], max_images: int) -> list[str]:
    if max_images <= 0:
        return []
    image_names: list[str] = []
    seen: set[str] = set()
    for result in results:
        for image_name, _ in image_positions_in_chunk(result.chunk):
            if image_name in seen:
                continue
            if find_image_path(image_name) is None:
                continue
            seen.add(image_name)
            image_names.append(image_name)
            if len(image_names) >= max_images:
                return image_names
    return image_names


def build_judge_prompt(
    question: str,
    product_name: str,
    results: list[RetrievalResult],
    image_names: list[str],
) -> str:
    image_labels = build_image_label_map(image_names)
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        chunk_image_names = [
            image_name
            for image_name, _ in image_positions_in_chunk(chunk)
            if image_name in image_labels
        ]
        blocks.append(
            "\n".join(
                [
                    f"[片段{index}] score={result.score:.4f} span={chunk.start}:{chunk.end}",
                    "图片标签：" + (
                        ", ".join(
                            f"<{image_labels[name]}: {name}>"
                            for name in chunk_image_names
                        )
                        if chunk_image_names
                        else "无"
                    ),
                    annotate_pic_tokens(chunk.text, chunk.image_names, image_labels),
                ]
            )
        )

    provided_images = "\n".join(
        f"<{image_labels[name]}: {name}>" for name in image_names
    ) or "无"

    return (
        "你是 RAG 召回覆盖度评估员。请判断给定手册片段是否足够回答用户问题。\n"
        "你会同时看到文本片段和插图。片段中的 <PIC_数字: 图片名> 表示该位置对应已提供图片。\n"
        "如果答案所需信息主要在图片中，只要图片已提供，也应视为覆盖。\n\n"
        "判定标准：\n"
        "- full：文本和/或图片已经覆盖回答问题所需的关键信息，可以直接回答。\n"
        "- partial：召回到了相关位置，但缺少关键步骤、条件、表格项、图片细节或上下文。\n"
        "- miss：片段基本无关，无法回答问题。\n\n"
        "只输出 JSON，不要输出 Markdown。\n"
        'JSON 格式：{"coverage":"full|partial|miss","reason":"一句话说明",'
        '"evidence":["关键证据1","关键证据2"],"used_images":["图片名"]}\n\n'
        f"产品：{product_name}\n"
        f"用户问题：{question}\n\n"
        f"已提供图片：\n{provided_images}\n\n"
        "召回片段：\n"
        + "\n\n".join(blocks)
    )


def build_multimodal_content(prompt: str, image_names: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    labels = build_image_label_map(image_names)
    for image_name in image_names:
        image_path = find_image_path(image_name)
        if image_path is None:
            continue
        content.append({"type": "text", "text": f"图片 <{labels[image_name]}: {image_name}>"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path_to_data_url(image_path)},
            }
        )
    return content


def is_data_inspection_error(exc: Exception) -> bool:
    text = str(exc).casefold()
    return (
        "datainspectionfailed" in text
        or "data_inspection_failed" in text
        or "input image data" in text
        or "inappropriate content" in text
    )


async def judge_one(
    client: AsyncOpenAI,
    model: str,
    case: dict[str, Any],
    top_k: int,
    retrieval_top_k: int,
    bm25_top_k: int,
    max_images: int,
    reranker: QwenReranker | None,
    vl_reranker: QwenVLReranker | None,
    rerank_instruction: str,
) -> dict[str, Any]:
    if not case.get("manual_path") or not Path(case["manual_path"]).exists():
        return {
            "id": case.get("id", ""),
            "question": case["question"],
            "product_name": case.get("product_name", ""),
            "manual_path": str(case.get("manual_path", "")),
            "coverage": "customer",
            "reason": "当前路由未进入产品专家，无法执行手册 RAG 覆盖判断。",
            "evidence": "[]",
            "used_images": "[]",
            "provided_images": "[]",
            "without_images": "0",
            "top_k": top_k,
            "retrieval_top_k": retrieval_top_k,
            "bm25_top_k": bm25_top_k,
            "rerank_scores": "",
            "chunk_count": 0,
            "top_scores": "",
            "raw": "",
            "original_product_name": case.get("original_product_name", ""),
            "original_manual_path": case.get("original_manual_path", ""),
            "reroute_agent_type": case.get("reroute_agent_type", ""),
            "reroute_reason": case.get("reroute_reason", ""),
        }

    candidate_results = get_default_rag().retrieve(
        question=case["question"],
        manual_path=case["manual_path"],
        product_name=case["product_name"],
        top_k=retrieval_top_k,
    )
    if bm25_top_k > 0:
        candidate_results = merge_retrieval_results(
            candidate_results,
            bm25_retrieve(
                question=case["question"],
                manual_path=case["manual_path"],
                product_name=case["product_name"],
                top_k=bm25_top_k,
            ),
        )
    rerank_scores = ""
    if vl_reranker is not None:
        results, rerank_scores = vl_reranker.rerank(
            question=case["question"],
            results=candidate_results,
            top_k=top_k,
            instruction=rerank_instruction,
        )
    elif reranker is None:
        results = candidate_results[:top_k]
    else:
        results, rerank_scores = rerank_results(
            reranker=reranker,
            question=case["question"],
            results=candidate_results,
            top_k=top_k,
            instruction=rerank_instruction,
        )
    image_names = collect_image_names(results, max_images)
    prompt = build_judge_prompt(case["question"], case["product_name"], results, image_names)
    without_images = False
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严格的多模态 RAG 召回覆盖度评估员。",
                },
                {
                    "role": "user",
                    "content": build_multimodal_content(prompt, image_names),
                },
            ],
            temperature=0,
        )
    except Exception as exc:
        if not image_names or not is_data_inspection_error(exc):
            raise
        without_images = True
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严格的 RAG 召回覆盖度评估员。",
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                        + "\n\n注意：图片因平台安全审核未能随消息发送，"
                        "请仅根据文本和图片标签判断；如果关键信息可能只在图片中，"
                        "应倾向判为 partial，而不是 error。"
                    ),
                },
            ],
            temperature=0,
        )
    raw = response.choices[0].message.content or ""
    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict):
        parsed = {"coverage": "parse_error", "reason": raw[:300], "evidence": [], "used_images": []}

    return {
        "id": case.get("id", ""),
        "question": case["question"],
        "product_name": case["product_name"],
        "manual_path": str(case["manual_path"]),
        "coverage": parsed.get("coverage", ""),
        "reason": parsed.get("reason", ""),
        "evidence": json.dumps(parsed.get("evidence", []), ensure_ascii=False),
        "used_images": json.dumps(parsed.get("used_images", []), ensure_ascii=False),
        "provided_images": json.dumps(image_names, ensure_ascii=False),
        "without_images": "1" if without_images else "0",
        "top_k": top_k,
        "retrieval_top_k": retrieval_top_k,
        "bm25_top_k": bm25_top_k,
        "rerank_scores": rerank_scores,
        "chunk_count": len(results),
        "top_scores": "|".join(f"{result.score:.6f}" for result in results),
        "raw": raw,
        "original_product_name": case.get("original_product_name", ""),
        "original_manual_path": case.get("original_manual_path", ""),
        "reroute_agent_type": case.get("reroute_agent_type", ""),
        "reroute_reason": case.get("reroute_reason", ""),
    }


def output_fieldnames() -> list[str]:
    return [
        "id",
        "question",
        "product_name",
        "manual_path",
        "coverage",
        "reason",
        "evidence",
        "used_images",
        "provided_images",
        "without_images",
        "top_k",
        "retrieval_top_k",
        "bm25_top_k",
        "rerank_scores",
        "chunk_count",
        "top_scores",
        "raw",
        "original_product_name",
        "original_manual_path",
        "reroute_agent_type",
        "reroute_reason",
    ]


async def run(args: argparse.Namespace) -> None:
    config = load_config()
    agent_config = get_agent_config(config, "expert")
    api_key = agent_config.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("缺少 api_key，请在 config.yaml 或 DASHSCOPE_API_KEY 中配置。")

    model = args.model or agent_config.model
    client = AsyncOpenAI(api_key=api_key, base_url=agent_config.base_url)
    reranker = None
    vl_reranker = None
    if args.rerank:
        reranker = QwenReranker(
            args.reranker_dir,
            max_length=args.reranker_max_length,
            batch_size=args.reranker_batch_size,
        )
    if args.vl_rerank:
        vl_reranker = QwenVLReranker(
            args.vl_reranker_dir,
            max_images_per_chunk=args.vl_images_per_chunk,
            max_chars=args.vl_reranker_max_chars,
        )
    retrieval_top_k = args.retrieval_top_k or args.top_k
    cases = (
        load_cases_from_csv(args.input_csv)
        if args.input_csv
        else load_cases_from_expert_log(args.input_log)
    )
    if args.reroute:
        cases = await reroute_cases(cases)
    if args.limit:
        cases = cases[: args.limit]

    rows: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                return await judge_one(
                    client=client,
                    model=model,
                    case=case,
                    top_k=args.top_k,
                    retrieval_top_k=retrieval_top_k,
                    bm25_top_k=args.bm25_top_k,
                    max_images=args.max_images,
                    reranker=reranker,
                    vl_reranker=vl_reranker,
                    rerank_instruction=args.rerank_instruction,
                )
            except Exception as exc:
                return {
                    "id": case.get("id", ""),
                    "question": case.get("question", ""),
                    "product_name": case.get("product_name", ""),
                    "manual_path": str(case.get("manual_path", "")),
                    "coverage": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "evidence": "[]",
                    "used_images": "[]",
                    "provided_images": "[]",
                    "without_images": "0",
                    "top_k": args.top_k,
                    "retrieval_top_k": retrieval_top_k,
                    "bm25_top_k": args.bm25_top_k,
                    "rerank_scores": "",
                    "chunk_count": 0,
                    "top_scores": "",
                    "raw": "",
                    "original_product_name": case.get("original_product_name", ""),
                    "original_manual_path": case.get("original_manual_path", ""),
                    "reroute_agent_type": case.get("reroute_agent_type", ""),
                    "reroute_reason": case.get("reroute_reason", ""),
                }

    fieldnames = output_fieldnames()
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, task in enumerate(
            asyncio.as_completed([guarded(case) for case in cases]),
            start=1,
        ):
            row = await task
            rows.append(row)
            writer.writerow(row)
            file.flush()
            print(
                f"[{index}/{len(cases)}] {row['coverage']} "
                f"{row['product_name']} images={len(json.loads(row['provided_images']))}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a multimodal LLM judge to evaluate whether RAG chunks cover each question."
    )
    parser.add_argument("--input-log", type=Path, default=DEFAULT_INPUT_LOG)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--retrieval-top-k", type=int, default=0)
    parser.add_argument("--bm25-top-k", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model", default="")
    parser.add_argument("--reroute", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--reranker-dir", type=Path, default=DEFAULT_RERANKER_DIR)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--rerank-instruction", default=DEFAULT_RERANK_INSTRUCTION)
    parser.add_argument("--vl-rerank", action="store_true")
    parser.add_argument("--vl-reranker-dir", type=Path, default=DEFAULT_VL_RERANKER_DIR)
    parser.add_argument("--vl-images-per-chunk", type=int, default=2)
    parser.add_argument("--vl-reranker-max-chars", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
