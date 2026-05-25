import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

from rag import (
    MANUAL_DIR as RAG_MANUAL_DIR,
    RetrievalResult,
    cosine_similarity,
    expand_with_neighbor_chunks,
    get_default_rag,
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

    async def _chat_with_history(
        self,
        history: List[ChatMessage],
        system_prompt: str,
        extra_user_context: str = "",
    ) -> str:
        completion = await self._client().chat.completions.create(
            model=self.model,
            messages=self._history_messages(history, system_prompt, extra_user_context),
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

        for manual_path in self._manual_files():
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
        if result.get("agent_type") != "expert":
            return None

        labels = [candidate["label"] for candidate in candidates]
        product_name = self._normalize_candidate_product_name(
            str(result.get("product_name") or ""),
            labels,
        )
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
                        "text": self._normalize_rag_text(result.chunk.text),
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

    def _normalize_rag_text(self, text: str) -> str:
        return " ".join(text.replace("<PIC>", " ").split())[: self.global_rag_max_chars]

    async def _classify_global_rag_candidates(
        self,
        question: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt = (
            "你是产品手册路由器。请根据用户问题和全局RAG候选片段判断是否应交给产品手册专家。"
            "只要候选片段能直接回答手册类问题，就选择对应产品手册；"
            "只有售后、物流、发票、退换货、投诉等非手册问题，或候选证据明显无关时，才返回customer。"
            "product_name必须精确使用候选手册名，不要附加分数或解释。只输出JSON。"
        )
        prompt_lines = [
            f"用户问题：{question}",
            "",
            "全局RAG候选手册与片段：",
        ]
        for index, candidate in enumerate(candidates, start=1):
            prompt_lines.append(
                f"{index}. {candidate['label']} "
                f"(avg={candidate['avg_score']:.4f}, max={candidate['max_score']:.4f})"
            )
            for snippet_index, snippet in enumerate(candidate["snippets"], start=1):
                prompt_lines.append(
                    f"   片段{snippet_index} score={snippet['score']:.4f}: {snippet['text']}"
                )
        prompt_lines.extend(
            [
                "",
                '输出JSON：{"agent_type":"expert或customer","product_name":"候选手册名或null","reason":"简短原因"}',
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
            )
        except Exception:
            return {}

        return parse_json_object(completion.choices[0].message.content or "")

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
        manual_options = self._manual_options_for_llm(rule_manual_path)
        if not manual_options:
            return None

        system_prompt = (
            "你是电商客服系统的产品手册路由器。"
            "请根据用户问题和每本手册摘要，判断问题是否应该交给某一本产品手册处理。"
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

    def _manual_options_for_llm(self, rule_manual_path: Path | None = None) -> str:
        manual_files = self._manual_files()
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
        agent_settings = load_config().get("agents", {}).get("expert", {})
        retrieval_settings = agent_settings.get("retrieval", {})
        self.retrieval_top_k = int(retrieval_settings.get("top_k", 6))
        self.retrieval_neighbor_count = int(retrieval_settings.get("neighbor_count", 1))

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

        image_names = unique_image_names(results)
        raw_content, candidate_image_names, error_text = await self._call_expert_model_with_retry(
            history=history,
            intent=intent,
            results=results,
            image_names=image_names,
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
        neighbor_count: int | None = None,
    ) -> List[RetrievalResult]:
        top_k = self.retrieval_top_k if top_k is None else top_k
        neighbor_count = self.retrieval_neighbor_count if neighbor_count is None else neighbor_count

        if intent.routing_results:
            top_results = sorted(
                intent.routing_results,
                key=lambda item: item.score,
                reverse=True,
            )[:top_k]
            chunks = [
                result.chunk
                for result in sorted(
                    intent.routing_results,
                    key=lambda item: (
                        item.chunk.entry_index,
                        item.chunk.start,
                        item.chunk.end,
                    ),
                )
            ]
            return expand_with_neighbor_chunks(top_results, chunks, neighbor_count)

        return get_default_rag().retrieve(
            question=question,
            manual_path=intent.manual_path,
            product_name=intent.product_name,
            top_k=top_k,
            neighbor_count=neighbor_count,
        )

    async def _call_expert_model_with_retry(
        self,
        *,
        history: List[ChatMessage],
        intent: IntentResult,
        results: List[RetrievalResult],
        image_names: List[str],
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
            )
            try:
                completion = await self._client().chat.completions.create(
                    model=self.model,
                    messages=messages,
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
    ) -> List[Dict[str, Any]]:
        system_prompt = self.prompt_template or (
            "你是产品手册专家助手，必须只根据给定手册片段回答。"
        )
        system_prompt = f"{system_prompt}\n{LANGUAGE_SYSTEM_RULE}"
        text_prompt = build_expert_prompt(history, intent, results, image_names)
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

