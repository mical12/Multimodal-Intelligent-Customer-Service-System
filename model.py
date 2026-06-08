import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

from agentic_rag import AgenticRAGConfig, run_agentic_rag
from rag import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL_DIR,
    MANUAL_DIR as RAG_MANUAL_DIR,
    RetrievalResult,
    cosine_similarity,
    get_default_rag,
    prepare_expert_context_results,
    product_name_from_manual,
    warmup_targets,
)
from util import (
    AgentConfig,
    ChatMessage,
    IntentResult,
    ManualContent,
    ManualMatch,
    MAX_EXPERT_IMAGES,
    annotate_pic_tokens,
    build_expert_prompt,
    build_image_label_map,
    build_extractive_expert_answer,
    find_image_path,
    get_agent_config,
    image_path_to_data_url,
    latest_user_message,
    load_config,
    load_manual_aliases,
    load_manual_entries,
    load_manual_sub_aliases,
    load_manual_summaries,
    normalize_answer_pic_tags,
    parse_expert_json,
    parse_json_object,
    record_expert_response,
    record_intent,
    unique_image_names,
    MANUAL_DIR,
)


FIXED_TEST_REPLY = "您好，您的问题已收到，我们会尽快为您处理。"
LLM_ROUTER_PRODUCTS = {""}
MAX_EXPERT_MODEL_RETRIES = 3
MAX_GLOBAL_RAG_IMAGES = 20
GLOBAL_RAG_INDEXES: list[dict[str, Any]] | None = None
LANGUAGE_SYSTEM_RULE = (
    "请始终根据用户最新问题的语言回答：中文问题必须用中文回答，"
    "英文问题必须用英文回答。不要因为手册片段、历史会话或示例的语言改变回答语言。"
)


class BaseAgent:
    """Base class for async LLM-backed agents."""

    def __init__(self, agent_config: AgentConfig):
        self.api_key = agent_config.api_key
        self.base_url = agent_config.base_url
        self.model = agent_config.model
        self.prompt_template = agent_config.prompt_template
        self._async_client: AsyncOpenAI | None = None

    def _client(self) -> AsyncOpenAI:
        if self._async_client is not None:
            return self._async_client

        api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("缺少 api_key，请在 config.yaml 或 DASHSCOPE_API_KEY 中配置。")

        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        return self._async_client

    def _history_messages(
        self,
        history: List[ChatMessage],
        system_prompt: str,
        extra_user_context: str = "",
    ) -> List[ChatMessage]:
        messages: List[ChatMessage] = [
            {"role": "system", "content": system_prompt},
        ]

        if extra_user_context:
            messages.append({"role": "user", "content": extra_user_context})

        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        return messages

    def _chat_completion_options(self, enable_thinking: bool | None = None) -> dict[str, Any]:
        model_name = str(self.model or "").lower()
        if "qwen3" in model_name:
            return {"extra_body": {"enable_thinking": bool(enable_thinking)}}
        return {}

    async def _chat_with_history(
        self,
        history: List[ChatMessage],
        system_prompt: str,
        extra_user_context: str = "",
    ) -> str:
        completion = await self._client().chat.completions.create(
            model=self.model,
            messages=self._history_messages(history, system_prompt, extra_user_context),
            **self._chat_completion_options(),
        )
        return completion.choices[0].message.content or ""



class IntentAgent(BaseAgent):
    """Decide whether a conversation should be handled by customer service or an expert."""

    def __init__(self, agent_config: AgentConfig, manual_dir: Path = MANUAL_DIR):
        super().__init__(agent_config)
        self.manual_dir = manual_dir
        self.manual_aliases = load_manual_aliases()
        self.manual_sub_aliases = load_manual_sub_aliases()
        self.manual_summaries = load_manual_summaries()
        agent_settings = load_config().get("agents", {}).get("intent", {})
        self.routing_mode = str(agent_settings.get("routing_mode", "summary")).strip().lower()
        rag_settings = agent_settings.get("global_rag_assist", {})
        self.global_rag_enabled = bool(rag_settings.get("enabled", False))
        self.global_rag_manual_count = int(rag_settings.get("manual_count", 10))
        self.global_rag_snippets_per_manual = int(rag_settings.get("snippets_per_manual", 2))
        self.global_rag_top_k = int(rag_settings.get("global_top_k", 80))
        self.global_rag_max_chars = int(rag_settings.get("snippet_max_chars", 420))
        self.global_rag_max_images = int(rag_settings.get("max_images", MAX_GLOBAL_RAG_IMAGES))

    async def recognize(self, history: List[ChatMessage]) -> IntentResult:
        question = latest_user_message(history)
        manual_match = self._match_manual(question)

        if manual_match is not None:
            manual_product_name = self._product_name_from_manual(manual_match.manual_path)
            if manual_product_name not in LLM_ROUTER_PRODUCTS:
                return IntentResult(
                    agent_type="expert",
                    product_name=manual_match.product_name,
                    manual_path=manual_match.manual_path,
                    manual_content=manual_match.manual_content,
                    image_names=manual_match.image_names,
                    reason="规则命中产品手册",
                    routing_results=manual_match.routing_results,
                )

        if self.routing_mode == "global_rag":
            rag_manual_match = await self._match_manual_with_global_rag(question)
            if rag_manual_match is not None:
                return IntentResult(
                    agent_type="expert",
                    product_name=rag_manual_match.product_name,
                    manual_path=rag_manual_match.manual_path,
                    manual_content=rag_manual_match.manual_content,
                    image_names=rag_manual_match.image_names,
                    reason="全局RAG候选片段辅助命中产品手册",
                    routing_results=rag_manual_match.routing_results,
                )
        else:
            llm_manual_match = await self._match_manual_with_llm(question, manual_match)
            if llm_manual_match is not None:
                return IntentResult(
                    agent_type="expert",
                    product_name=llm_manual_match.product_name,
                    manual_path=llm_manual_match.manual_path,
                    manual_content=llm_manual_match.manual_content,
                    image_names=llm_manual_match.image_names,
                    reason="大模型结合手册摘要命中产品手册",
                    routing_results=llm_manual_match.routing_results,
                )

        return IntentResult(
            agent_type="customer",
            reason=f"规则和{self.routing_mode}模式均未命中产品手册，走通用客服",
        )

    def _match_manual(self, question: str) -> ManualMatch | None:
        question_text = question.casefold()
        best_match: tuple[int, ManualMatch] | None = None

        for manual_path in self._manual_files_for_question(question):
            product_name = self._product_name_from_manual(manual_path)

            if product_name in self.manual_sub_aliases:
                for sub_product_name, sub_aliases in self.manual_sub_aliases[product_name].items():
                    aliases = [sub_product_name, *sub_aliases]
                    for alias in aliases:
                        alias_text = str(alias).strip().casefold()
                        if not alias_text or alias_text not in question_text:
                            continue
                        manual_content = self._manual_content_by_sub_product(
                            manual_path,
                            product_name,
                            sub_product_name,
                        )
                        if manual_content is None:
                            continue
                        manual_match = ManualMatch(
                            product_name=sub_product_name,
                            manual_path=manual_path,
                            manual_content=manual_content.content,
                            image_names=manual_content.image_names,
                        )
                        alias_length = len(alias_text)
                        if best_match is None or alias_length > best_match[0]:
                            best_match = (alias_length, manual_match)
                continue

            aliases = [product_name, *self.manual_aliases.get(product_name, [])]
            for alias in aliases:
                alias_text = str(alias).strip().casefold()
                if not alias_text or alias_text not in question_text:
                    continue
                manual_content = self._manual_content_by_index(manual_path, 0)
                if manual_content is None:
                    continue
                manual_match = ManualMatch(
                    product_name=product_name,
                    manual_path=manual_path,
                    manual_content=manual_content.content,
                    image_names=manual_content.image_names,
                )
                alias_length = len(alias_text)
                if best_match is None or alias_length > best_match[0]:
                    best_match = (alias_length, manual_match)

        return best_match[1] if best_match else None

    def _manual_content_by_sub_product(
        self,
        manual_path: Path,
        product_name: str,
        sub_product_name: str,
    ) -> ManualContent | None:
        sub_product_names = list(self.manual_sub_aliases.get(product_name, {}).keys())
        try:
            entry_index = sub_product_names.index(sub_product_name)
        except ValueError:
            return None

        return self._manual_content_by_index(manual_path, entry_index)

    def _manual_content_by_index(
        self,
        manual_path: Path,
        entry_index: int,
    ) -> ManualContent | None:
        entries = self._load_manual_entries(manual_path)
        if entry_index >= len(entries):
            return None

        return entries[entry_index]

    def _load_manual_entries(self, manual_path: Path) -> List[ManualContent]:
        return load_manual_entries(manual_path)

    async def warmup_global_rag(self) -> None:
        if self.routing_mode != "global_rag":
            return
        if not self.global_rag_enabled:
            return
        await asyncio.to_thread(self._global_rag_indexes)

    async def _match_manual_with_global_rag(self, question: str) -> ManualMatch | None:
        if not self.global_rag_enabled:
            return None
        if self._skip_global_rag_for_customer_question(question):
            return None

        candidates = await asyncio.to_thread(self._global_rag_candidates, question)
        if not candidates:
            return None

        result = await self._classify_global_rag_candidates(question, candidates)

        labels = self._global_rag_allowed_product_names(question, candidates)
        product_name = self._normalize_candidate_product_name(
            str(result.get("product_name") or ""),
            labels,
        )
        agent_type = str(result.get("agent_type") or "").strip().lower()
        if not product_name and agent_type != "expert":
            return None
        if product_name and agent_type == "customer":
            return None
        if not product_name:
            return None

        scored_results = []
        for candidate in candidates:
            if candidate["label"] == product_name:
                scored_results = candidate.get("scored_results", [])
                break

        return self._manual_match_for_product_name(product_name, scored_results)

    def _global_rag_candidates(self, question: str) -> list[dict[str, Any]]:
        all_results_by_label = self._global_score_by_label(question)
        results = sorted(
            (
                (label, result)
                for label, label_results in all_results_by_label.items()
                for result in label_results
            ),
            key=lambda item: item[1].score,
            reverse=True,
        )[: self.global_rag_top_k]
        by_label: dict[str, list[RetrievalResult]] = {}
        for label, result in results:
            by_label.setdefault(label, []).append(result)

        ranked = []
        for label, label_results in by_label.items():
            sorted_results = sorted(label_results, key=lambda item: item.score, reverse=True)
            top_scores = [item.score for item in sorted_results[:3]]
            if not top_scores:
                continue
            ranked.append(
                {
                    "label": label,
                    "max_score": max(top_scores),
                    "avg_score": sum(top_scores) / len(top_scores),
                    "results": sorted_results,
                }
            )

        ranked.sort(key=lambda item: (-item["avg_score"], -item["max_score"], item["label"]))
        candidates = []
        for item in ranked[: self.global_rag_manual_count]:
            snippets = []
            for result in item["results"][: self.global_rag_snippets_per_manual]:
                snippets.append(
                    {
                        "score": result.score,
                        "text": result.chunk.text,
                        "image_names": [
                            image_name
                            for image_name in result.chunk.image_names
                            if find_image_path(image_name) is not None
                        ],
                    }
                )
            candidates.append(
                {
                    "label": item["label"],
                    "max_score": item["max_score"],
                    "avg_score": item["avg_score"],
                    "snippets": snippets,
                    "scored_results": all_results_by_label.get(item["label"], []),
                }
            )
        return candidates

    def _global_retrieve(self, question: str, top_k: int) -> list[tuple[str, RetrievalResult]]:
        all_results_by_label = self._global_score_by_label(question)
        best = [
            (label, result)
            for label, label_results in all_results_by_label.items()
            for result in label_results
        ]
        return sorted(best, key=lambda item: item[1].score, reverse=True)[:top_k]

    def _global_score_by_label(self, question: str) -> dict[str, list[RetrievalResult]]:
        rag = get_default_rag()
        question_vector = rag._encode_query(question)
        by_label: dict[str, list[RetrievalResult]] = {}

        for index in self._global_rag_indexes():
            if not self._index_matches_question_language(index, question):
                continue
            best_by_chunk: dict[int, RetrievalResult] = {}
            for chunk_index, document_vector in zip(
                index["document_chunk_indexes"],
                index["document_vectors"],
            ):
                chunk = index["chunks"][chunk_index]
                score = cosine_similarity(question_vector, document_vector)
                current = best_by_chunk.get(chunk_index)
                if current is None or score > current.score:
                    best_by_chunk[chunk_index] = RetrievalResult(chunk=chunk, score=score)
            by_label[index["label"]] = sorted(
                best_by_chunk.values(),
                key=lambda item: item.score,
                reverse=True,
            )

        return by_label

    @staticmethod
    def _global_rag_indexes() -> list[dict[str, Any]]:
        global GLOBAL_RAG_INDEXES
        if GLOBAL_RAG_INDEXES is not None:
            return GLOBAL_RAG_INDEXES

        rag = get_default_rag()
        indexes: list[dict[str, Any]] = []
        for manual_path, product_name in warmup_targets(RAG_MANUAL_DIR):
            index = rag.load_or_build_index(manual_path, product_name=product_name)
            indexes.append(
                {
                    "manual_path": manual_path,
                    "product_name": product_name,
                    "label": product_name or product_name_from_manual(manual_path),
                    "chunks": index["chunks"],
                    "document_chunk_indexes": index["document_chunk_indexes"],
                    "document_vectors": index["document_vectors"],
                }
            )
        GLOBAL_RAG_INDEXES = indexes
        return indexes

    def _normalize_rag_text(
        self,
        text: str,
        image_names: list[str] | None = None,
        image_labels: dict[str, str] | None = None,
    ) -> str:
        image_names = image_names or []
        image_labels = image_labels or {}
        if image_names and image_labels:
            text = annotate_pic_tokens(text, image_names, image_labels)
        else:
            text = text.replace("<PIC>", " ")
        return " ".join(text.split())[: self.global_rag_max_chars]

    async def _classify_global_rag_candidates(
        self,
        question: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt = (
            "你是产品手册路由器。请根据用户问题和全局RAG候选片段判断是否应交给产品手册专家。"
            "候选片段中的 <PIC_数字: 图片名> 表示该位置有对应插图；如果图片已提供，请结合图片判断。"
            "请同时参考手册摘要；即使RAG片段没有直接命中，只要用户问的是某个具体产品手册中的安装、使用、"
            "设置、维护、故障排查、安全注意事项等问题，也应选择对应产品手册。"
            "只有售后、物流、发票、退换货、投诉等非手册问题，或候选片段和手册摘要都明显无关时，才返回customer。"
            "agent_type只能是expert或customer二选一；只要product_name不是null，agent_type必须为expert。"
            "product_name必须精确使用候选手册名或摘要中的产品名，不要附加分数或解释。只输出JSON。"
        )
        image_names = (
            self._global_rag_candidate_image_names(candidates)
            if self._model_supports_image_input()
            else []
        )
        image_labels = build_image_label_map(image_names)
        summary_options = self._global_rag_summary_options(question)
        prompt_lines = [
            f"用户问题：{question}",
            "",
            "已提供图片：",
            "\n".join(
                f"<{image_labels[image_name]}: {image_name}>"
                for image_name in image_names
            ) if image_names else "无",
            "",
            "同语言手册摘要：",
            summary_options or "无",
            "",
            "全局RAG候选手册与片段：",
        ]
        for index, candidate in enumerate(candidates, start=1):
            prompt_lines.append(
                f"{index}. {candidate['label']} "
                f"(avg={candidate['avg_score']:.4f}, max={candidate['max_score']:.4f})"
            )
            for snippet_index, snippet in enumerate(candidate["snippets"], start=1):
                snippet_image_names = [
                    image_name
                    for image_name in snippet.get("image_names", [])
                    if image_name in image_labels
                ]
                prompt_lines.append(
                    f"   片段{snippet_index} score={snippet['score']:.4f}: "
                    f"{self._normalize_rag_text(snippet['text'], snippet_image_names, image_labels)}"
                )
        prompt_lines.extend(
            [
                "",
                '输出JSON：{"agent_type":"expert或customer","product_name":"候选手册名或null","reason":"简短原因"}',
            ]
        )

        try:
            user_content = self._global_rag_user_content("\n".join(prompt_lines), image_names)
            completion = await self._client().chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                **self._chat_completion_options(),
            )
        except Exception as exc:
            if not image_names or not self._is_data_inspection_error(exc):
                return {}
            prompt_lines.extend(
                [
                    "",
                    "注意：图片因平台安全审核未能随消息发送，请仅根据文本和图片标签判断。",
                ]
            )
            try:
                completion = await self._client().chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "\n".join(prompt_lines)},
                    ],
                    **self._chat_completion_options(),
                )
            except Exception:
                return {}

        return parse_json_object(completion.choices[0].message.content or "")

    def _global_rag_candidate_image_names(self, candidates: list[dict[str, Any]]) -> list[str]:
        if self.global_rag_max_images <= 0:
            return []

        image_names: list[str] = []
        seen = set()
        for candidate in candidates:
            for snippet in candidate.get("snippets", []):
                for image_name in snippet.get("image_names", []):
                    if image_name in seen:
                        continue
                    if find_image_path(image_name) is None:
                        continue
                    seen.add(image_name)
                    image_names.append(image_name)
                    if len(image_names) >= self.global_rag_max_images:
                        return image_names
        return image_names

    @staticmethod
    def _global_rag_user_content(
        prompt: str,
        image_names: list[str],
    ) -> str | list[dict[str, Any]]:
        if not image_names:
            return prompt

        image_labels = build_image_label_map(image_names)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_name in image_names:
            image_path = find_image_path(image_name)
            if image_path is None:
                continue
            content.append({"type": "text", "text": f"图片 <{image_labels[image_name]}: {image_name}>"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path_to_data_url(image_path),
                    },
                }
            )
        return content

    def _global_rag_summary_options(self, question: str) -> str:
        language = self._question_language(question)
        if language == "en":
            lines = []
            summary = self.manual_summaries.get("英文汇总")
            if summary:
                lines.append(f"- 英文汇总：{summary}")
            sub_products = list(self.manual_sub_aliases.get("英文汇总", {}).keys())
            if sub_products:
                lines.append("英文汇总中的可选英文产品：" + "、".join(sub_products))
            return "\n".join(lines)

        lines = []
        for manual_path in self._manual_files_for_question(question):
            product_name = self._product_name_from_manual(manual_path)
            summary = self.manual_summaries.get(product_name)
            if summary:
                lines.append(f"- {product_name}：{summary}")
        return "\n".join(lines)

    def _global_rag_allowed_product_names(
        self,
        question: str,
        candidates: list[dict[str, Any]],
    ) -> list[str]:
        names = [candidate["label"] for candidate in candidates]
        if self._question_language(question) == "en":
            names.extend(self.manual_sub_aliases.get("英文汇总", {}).keys())
            names.append("英文汇总")
        else:
            names.extend(
                self._product_name_from_manual(path)
                for path in self._manual_files_for_question(question)
            )

        unique_names = []
        seen = set()
        for name in names:
            if name and name not in seen:
                seen.add(name)
                unique_names.append(name)
        return unique_names

    def _model_supports_image_input(self) -> bool:
        model_name = str(self.model or "").lower()
        return "vl" in model_name or "vision" in model_name

    @staticmethod
    def _is_data_inspection_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".casefold()
        return (
            "datainspectionfailed" in text
            or "data_inspection_failed" in text
            or "input image data" in text
            or "unknown variant `image_url`" in text
            or "unknown variant 'image_url'" in text
            or "expected `text`" in text
            or "expected 'text'" in text
            or "unexpected item type in content" in text
            or "inappropriate content" in text
        )

    @staticmethod
    def _normalize_candidate_product_name(product_name: str, labels: list[str]) -> str:
        name = str(product_name or "").strip()
        if not name or name.lower() == "null":
            return ""
        for label in labels:
            if name == label:
                return label
        for label in sorted(labels, key=len, reverse=True):
            if label and label in name:
                return label
        return ""

    def _manual_match_for_product_name(
        self,
        product_name: str,
        routing_results: list[RetrievalResult] | None = None,
    ) -> ManualMatch | None:
        for manual_path in self._manual_files():
            manual_name = self._product_name_from_manual(manual_path)
            if product_name == manual_name:
                manual_content = self._manual_content_by_index(manual_path, 0)
                if manual_content is None:
                    return None
                return ManualMatch(
                    product_name=product_name,
                    manual_path=manual_path,
                    manual_content=manual_content.content,
                    image_names=manual_content.image_names,
                    routing_results=routing_results or [],
                )

            if product_name in self.manual_sub_aliases.get(manual_name, {}):
                manual_content = self._manual_content_by_sub_product(
                    manual_path,
                    manual_name,
                    product_name,
                )
                if manual_content is None:
                    return None
                return ManualMatch(
                    product_name=product_name,
                    manual_path=manual_path,
                    manual_content=manual_content.content,
                    image_names=manual_content.image_names,
                    routing_results=routing_results or [],
                )

        return None

    @staticmethod
    def _skip_global_rag_for_customer_question(question: str) -> bool:
        text = question.casefold()
        customer_markers = [
            "纸质版说明书",
            "电子版在哪里",
            "电子版说明书",
            "生产日期",
            "出厂日期",
            "保修",
            "终身维修",
            "维修服务",
            "售后",
            "物流",
            "快递",
            "运费",
            "发票",
            "退货",
            "换货",
            "退款",
            "投诉",
            "订单",
            "库存",
            "价格",
        ]
        return any(marker in text for marker in customer_markers)

    async def _match_manual_with_llm(
        self,
        question: str,
        rule_manual_match: ManualMatch | None = None,
    ) -> ManualMatch | None:
        rule_manual_path = rule_manual_match.manual_path if rule_manual_match else None
        manual_options = self._manual_options_for_llm(question, rule_manual_path)
        if not manual_options:
            return None

        system_prompt = (
            "你是电商客服系统的产品手册路由器。"
            "请根据用户问题和每本手册摘要，判断问题是否应该交给某一本产品手册处理。"
            "英文问题只能在英文汇总手册的英文产品中选择；中文问题只能在中文单产品手册中选择。"
            "如果问题属于英文汇总手册中的某个产品，manual_name 必须返回英文汇总，"
            "product_name 必须返回英文汇总手册中对应产品的准确名称。"
            "只输出 JSON，不要输出解释。"
        )
        user_prompt = (
            f"用户问题：{question}\n\n"
            "可选产品手册摘要：\n"
            f"{manual_options}\n\n"
            "英文汇总中的 product_name 只能从这些名称中选择："
            f"{', '.join(self.manual_sub_aliases.get('英文汇总', {}).keys())}\n\n"
            '输出格式：{"agent_type":"expert或customer","manual_name":"手册名或null",'
            '"product_name":"产品名或null"}'
        )

        try:
            completion = await self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **self._chat_completion_options(),
            )
            content = completion.choices[0].message.content or ""
            result = self._parse_llm_router_result(content)
        except Exception:
            return None

        if result.get("agent_type") != "expert":
            return None

        manual_name = str(result.get("manual_name") or "").strip()
        product_name = str(result.get("product_name") or "").strip()
        if not manual_name and not product_name:
            return None

        if not manual_name and product_name in self.manual_sub_aliases.get("英文汇总", {}):
            manual_name = "英文汇总"

        if manual_name == "英文汇总":
            if not product_name:
                return None
            manual_path = self._manual_path_by_product_name(manual_name)
            if manual_path is None:
                return None
            manual_content = self._manual_content_by_sub_product(
                manual_path,
                manual_name,
                product_name,
            )
            if manual_content is None:
                return None
            return ManualMatch(
                product_name=product_name,
                manual_path=manual_path,
                manual_content=manual_content.content,
                image_names=manual_content.image_names,
            )

        lookup_name = manual_name or product_name
        manual_path = self._manual_path_by_product_name(lookup_name)
        if manual_path is None and product_name:
            manual_path = self._manual_path_by_product_name(product_name)
            lookup_name = product_name
        if manual_path is None:
            return None

        manual_content = self._manual_content_by_index(manual_path, 0)
        if manual_content is None:
            return None

        return ManualMatch(
            product_name=product_name or lookup_name,
            manual_path=manual_path,
            manual_content=manual_content.content,
            image_names=manual_content.image_names,
        )

    def _manual_options_for_llm(
        self,
        question: str,
        rule_manual_path: Path | None = None,
    ) -> str:
        manual_files = self._manual_files_for_question(question)
        products = [self._product_name_from_manual(path) for path in manual_files]

        if rule_manual_path is not None:
            rule_product = self._product_name_from_manual(rule_manual_path)
            if rule_product in LLM_ROUTER_PRODUCTS:
                products = [
                    product_name
                    for product_name in products
                    if product_name in LLM_ROUTER_PRODUCTS
                ]

        lines = []
        for product_name in products:
            summary = self.manual_summaries.get(product_name)
            if summary:
                lines.append(f"- {product_name}：{summary}")

        return "\n".join(lines)

    def _manual_files_for_question(self, question: str) -> List[Path]:
        language = self._question_language(question)
        manual_files = self._manual_files()
        if language == "en":
            return [
                path
                for path in manual_files
                if self._product_name_from_manual(path) == "英文汇总"
            ]
        if language == "zh":
            return [
                path
                for path in manual_files
                if self._product_name_from_manual(path) != "英文汇总"
            ]
        return manual_files

    def _index_matches_question_language(self, index: dict[str, Any], question: str) -> bool:
        language = self._question_language(question)
        manual_name = self._product_name_from_manual(Path(index["manual_path"]))
        if language == "en":
            return manual_name == "英文汇总"
        if language == "zh":
            return manual_name != "英文汇总"
        return True

    @staticmethod
    def _question_language(question: str) -> str:
        text = str(question or "")
        has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
        has_latin = any(char.isascii() and char.isalpha() for char in text)
        if has_cjk:
            return "zh"
        if has_latin:
            return "en"
        return "zh"

    @staticmethod
    def _parse_llm_router_result(content: str) -> Dict[str, Any]:
        return parse_json_object(content)

    def _manual_path_by_product_name(self, product_name: str) -> Path | None:
        for manual_path in self._manual_files():
            if self._product_name_from_manual(manual_path) == product_name:
                return manual_path

        return None

    def _manual_files(self) -> List[Path]:
        if not self.manual_dir.exists():
            return []

        return [
            path
            for path in self.manual_dir.glob("*手册.txt")
        ]

    @staticmethod
    def _product_name_from_manual(manual_path: Path) -> str:
        return manual_path.stem.removesuffix("手册")


class ExpertAgent(BaseAgent):
    """Handle product manual questions with RAG context and matched images."""

    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)
        config = load_config()
        agent_settings = config.get("agents", {}).get("expert", {})
        self.agentic_rag_config = AgenticRAGConfig.from_config(config, "expert")
        self.final_enable_thinking = bool(agent_settings.get("enable_thinking", False))
        retrieval_settings = agent_settings.get("retrieval", {})
        self.retrieval_top_k = int(retrieval_settings.get("top_k", 6))
        self.retrieval_neighbor_count = int(retrieval_settings.get("neighbor_count", 0))
        self.retrieval_candidate_top_k = int(
            retrieval_settings.get("candidate_top_k", self.retrieval_top_k)
        )
        self.retrieval_bm25_top_k = int(retrieval_settings.get("bm25_top_k", 0))
        rerank_settings = retrieval_settings.get("rerank", {})
        self.rerank_enabled = bool(rerank_settings.get("enabled", False))
        self.reranker_model_dir = Path(
            rerank_settings.get("model_dir") or DEFAULT_RERANKER_MODEL_DIR
        )
        self.reranker_max_length = int(
            rerank_settings.get("max_length", DEFAULT_RERANKER_MAX_LENGTH)
        )
        self.reranker_batch_size = int(
            rerank_settings.get("batch_size", DEFAULT_RERANKER_BATCH_SIZE)
        )

    async def reply(self, history: List[ChatMessage], intent: IntentResult) -> str:
        if intent.manual_path is None:
            return await self._fallback_expert_reply(history, intent)

        question = latest_user_message(history)
        results = await asyncio.to_thread(
            self._retrieve_manual_context,
            question,
            intent,
        )
        if not results:
            return await self._fallback_expert_reply(history, intent)

        evidence_summary = ""
        if self.agentic_rag_config.enabled:
            agentic_result = await run_agentic_rag(
                agent=self,
                question=question,
                intent=intent,
                initial_results=results,
                top_k=self.retrieval_top_k,
                config=self.agentic_rag_config,
            )
            if agentic_result.results:
                results = agentic_result.results
            evidence_summary = agentic_result.evidence_summary

        image_names = unique_image_names(results)
        raw_content, candidate_image_names, error_text = await self._call_expert_model_with_retry(
            history=history,
            intent=intent,
            results=results,
            image_names=image_names,
            evidence_summary=evidence_summary,
        )
        if not raw_content.strip():
            if error_text:
                print(f"expert model call failed: {error_text}")
            final_answer = build_extractive_expert_answer(results)
            final_image_names = candidate_image_names[:MAX_EXPERT_IMAGES]
            record_expert_response(
                history=history,
                intent=intent,
                model=self.model,
                retrieval_results=results,
                candidate_image_names=candidate_image_names,
                raw_content=error_text or "Empty expert response",
                parsed_answer="",
                parsed_image_names=[],
                final_answer=final_answer,
                final_image_names=final_image_names,
                user_id=None,
            )
            return self._format_expert_reply(final_answer, final_image_names)

        answer, parsed_images = parse_expert_json(raw_content)
        answer = normalize_answer_pic_tags(answer)
        selected_images = [
            image_name
            for image_name in parsed_images
            if image_name in image_names
        ]
        pic_count = answer.count("<PIC>")
        if pic_count <= 0:
            selected_images = []
        elif selected_images:
            selected_images = selected_images[:pic_count]
        else:
            selected_images = image_names[:pic_count]

        record_expert_response(
            history=history,
            intent=intent,
            model=self.model,
            retrieval_results=results,
            candidate_image_names=image_names,
            raw_content=raw_content,
            parsed_answer=answer,
            parsed_image_names=parsed_images,
            final_answer=answer,
            final_image_names=selected_images,
            user_id=None,
        )
        return self._format_expert_reply(answer, selected_images)

    def _retrieve_manual_context(
        self,
        question: str,
        intent: IntentResult,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        top_k = self.retrieval_top_k if top_k is None else top_k

        if self.rerank_enabled:
            results = get_default_rag().retrieve_with_bm25_rerank(
                question=question,
                manual_path=intent.manual_path,
                product_name=intent.product_name,
                embedding_top_k=self.retrieval_candidate_top_k,
                bm25_top_k=self.retrieval_bm25_top_k,
                final_top_k=top_k,
                neighbor_count=self.retrieval_neighbor_count,
                reranker_model_dir=self.reranker_model_dir,
                reranker_max_length=self.reranker_max_length,
                reranker_batch_size=self.reranker_batch_size,
            )
            index = get_default_rag().load_or_build_index(
                manual_path=intent.manual_path,
                product_name=intent.product_name,
            )
            return prepare_expert_context_results(
                results,
                index.get("chunks", []),
                max_results=top_k,
            )

        results = get_default_rag().retrieve(
            question=question,
            manual_path=intent.manual_path,
            product_name=intent.product_name,
            top_k=top_k,
            neighbor_count=self.retrieval_neighbor_count,
        )
        index = get_default_rag().load_or_build_index(
            manual_path=intent.manual_path,
            product_name=intent.product_name,
        )
        return prepare_expert_context_results(
            results,
            index.get("chunks", []),
            max_results=top_k,
        )

    async def _call_expert_model_with_retry(
        self,
        *,
        history: List[ChatMessage],
        intent: IntentResult,
        results: List[RetrievalResult],
        image_names: List[str],
        evidence_summary: str = "",
    ) -> tuple[str, List[str], str]:
        candidate_image_names = image_names
        last_error = ""
        without_images = False

        for attempt in range(1, MAX_EXPERT_MODEL_RETRIES + 1):
            candidate_image_names = [] if without_images else image_names
            messages = self._expert_messages(
                history,
                intent,
                results,
                candidate_image_names,
                evidence_summary=evidence_summary,
            )
            try:
                completion = await self._client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self._chat_completion_options(
                        enable_thinking=self.final_enable_thinking,
                    ),
                )
                raw_content = completion.choices[0].message.content or ""
                if raw_content.strip():
                    return raw_content, candidate_image_names, ""

                last_error = "Empty expert response"
                print(
                    "expert model returned empty content: "
                    f"attempt={attempt}/{MAX_EXPERT_MODEL_RETRIES} "
                    f"without_images={without_images}"
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if self._is_data_inspection_error(exc) and not without_images:
                    print(
                        "expert image data inspection failed; "
                        "retrying without images"
                    )
                    without_images = True
                    continue

                if self._is_retryable_expert_error(exc):
                    print(
                        "expert model retryable call failed: "
                        f"attempt={attempt}/{MAX_EXPERT_MODEL_RETRIES} "
                        f"without_images={without_images} {last_error}"
                    )
                else:
                    return "", candidate_image_names, last_error

            if attempt < MAX_EXPERT_MODEL_RETRIES:
                await asyncio.sleep(attempt)

        return "", candidate_image_names, last_error

    @staticmethod
    def _is_data_inspection_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".casefold()
        return (
            "datainspectionfailed" in text
            or "data_inspection_failed" in text
            or "input image data" in text
            or "unknown variant `image_url`" in text
            or "unknown variant 'image_url'" in text
            or "expected `text`" in text
            or "expected 'text'" in text
            or "inappropriate content" in text
        )

    @staticmethod
    def _is_retryable_expert_error(exc: Exception) -> bool:
        name = type(exc).__name__.casefold()
        text = str(exc).casefold()
        return (
            "connection" in name
            or "timeout" in name
            or "connect" in text
            or "timed out" in text
        )

    def _expert_messages(
        self,
        history: List[ChatMessage],
        intent: IntentResult,
        results: List[RetrievalResult],
        image_names: List[str],
        evidence_summary: str = "",
    ) -> List[Dict[str, Any]]:
        system_prompt = self.prompt_template or (
            "你是产品手册专家助手，必须只根据给定手册片段回答。"
        )
        system_prompt = f"{system_prompt}\n{LANGUAGE_SYSTEM_RULE}"
        text_prompt = build_expert_prompt(history, intent, results, image_names)
        if evidence_summary.strip():
            text_prompt += (
                "\n\nAgentic RAG has already summarized the useful evidence below. "
                "Use it to avoid missing key information, but still verify against the original chunks. "
                "Write the final answer in a natural, polite customer-service style. "
                "Do not copy the manual verbatim; explain the procedure or conclusion fluently. "
                "When an image is helpful, insert <PIC> at the relevant sentence and include the matching image name in JSON.\n"
                f"{evidence_summary.strip()}"
            )
        if not image_names:
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
            ]
            for message in history[:-1]:
                role = message.get("role", "")
                previous_content = message.get("content", "")
                if role in {"user", "assistant"} and previous_content:
                    messages.append({"role": role, "content": previous_content})
            messages.append({"role": "user", "content": text_prompt})
            return messages

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": text_prompt},
        ]
        image_labels = build_image_label_map(image_names)

        for image_name in image_names[:MAX_EXPERT_IMAGES]:
            image_path = find_image_path(image_name)
            if image_path is None:
                continue
            image_label = image_labels.get(image_name, "PIC")
            content.append({"type": "text", "text": f"图片 <{image_label}: {image_name}>"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path_to_data_url(image_path),
                    },
                }
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        for message in history[:-1]:
            role = message.get("role", "")
            previous_content = message.get("content", "")
            if role in {"user", "assistant"} and previous_content:
                messages.append({"role": role, "content": previous_content})
        messages.append({"role": "user", "content": content})
        return messages

    async def _fallback_expert_reply(
        self,
        history: List[ChatMessage],
        intent: IntentResult,
    ) -> str:
        system_prompt = self.prompt_template or "你是产品手册专家助手。"
        system_prompt = f"{system_prompt}\n{LANGUAGE_SYSTEM_RULE}"
        extra_user_context = (
            f"产品：{intent.product_name or '相关产品'}\n"
            "当前没有可用的手册检索结果，请基于历史会话给出谨慎、简洁的回复。"
        )
        answer = await self._chat_with_history(history, system_prompt, extra_user_context)
        return self._format_expert_reply(answer, [])

    @staticmethod
    def _format_expert_reply(answer: str, image_names: List[str]) -> str:
        answer = (answer or "").strip()
        if not answer:
            answer = "请参考以下手册片段处理该问题。"
        return f"{json.dumps(answer, ensure_ascii=False)}, {json.dumps(image_names, ensure_ascii=False)}"

class CustomerAgent(BaseAgent):
    """Handle general customer-service questions."""

    async def reply(self, history: List[ChatMessage], intent: IntentResult) -> str:
        system_prompt = self.prompt_template or "你是一个专业、耐心的电商客服助手。"
        system_prompt = f"{system_prompt}\n{LANGUAGE_SYSTEM_RULE}"
        extra_user_context = (
            "路由结果：电商客服\n"
            f"路由原因：{intent.reason}\n\n"
            "请结合完整历史会话，给出简洁、礼貌、可执行的客服回复。"
            "不要机械复述示例，需要根据用户当前问题作答。"
        )
        return await self._chat_with_history(history, system_prompt, extra_user_context)


config = load_config()
intent_agent = IntentAgent(get_agent_config(config, "intent"))
expert_agent = ExpertAgent(get_agent_config(config, "expert"))
customer_agent = CustomerAgent(get_agent_config(config, "customer"))


async def generate_reply(
    history: List[ChatMessage],
    user_id: str | None = None,
) -> str:
    intent = await intent_agent.recognize(history)
    record_intent(history, intent, user_id)
    if intent.agent_type == "expert":
       return await expert_agent.reply(history, intent)

    return await customer_agent.reply(history, intent)

