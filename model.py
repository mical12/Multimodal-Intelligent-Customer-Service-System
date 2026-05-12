import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

from rag import RetrievalResult, get_default_rag
from util import (
    AgentConfig,
    ChatMessage,
    IntentResult,
    ManualContent,
    ManualMatch,
    MAX_EXPERT_IMAGES,
    build_expert_prompt,
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
    parse_expert_json,
    parse_json_object,
    record_intent,
    unique_image_names,
    MANUAL_DIR,
)


FIXED_TEST_REPLY = "您好，您的问题已收到，我们会尽快为您处理。"
LLM_ROUTER_PRODUCTS = {""}


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
                )

        llm_manual_match = await self._match_manual_with_llm(question, manual_match)
        if llm_manual_match is not None:
            return IntentResult(
                agent_type="expert",
                product_name=llm_manual_match.product_name,
                manual_path=llm_manual_match.manual_path,
                manual_content=llm_manual_match.manual_content,
                image_names=llm_manual_match.image_names,
                reason="大模型结合手册摘要命中产品手册",
            )

        return IntentResult(
            agent_type="customer",
            reason="规则和大模型均未命中产品手册，走通用客服",
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
            "如果问题是售后、物流、发票、退换货、投诉等通用客服问题，问题没有明确产品线索，即使提到 troubleshooting、safety、maintenance，返回 customer。"
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
        messages = self._expert_messages(history, intent, results, image_names)

        try:
            completion = await self._client().chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = completion.choices[0].message.content or ""
        except Exception:
            return self._format_expert_reply(
                build_extractive_expert_answer(results),
                image_names[:MAX_EXPERT_IMAGES],
            )

        answer, selected_images = parse_expert_json(content)
        selected_images = [
            image_name
            for image_name in selected_images
            if image_name in image_names
        ]
        if not selected_images:
            selected_images = image_names[:MAX_EXPERT_IMAGES]

        return self._format_expert_reply(answer, selected_images)

    def _retrieve_manual_context(
        self,
        question: str,
        intent: IntentResult,
    ) -> List[RetrievalResult]:
        return get_default_rag().retrieve(
            question=question,
            manual_path=intent.manual_path,
            product_name=intent.product_name,
            top_k=3,
            neighbor_count=1,
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
        text_prompt = build_expert_prompt(history, intent, results, image_names)
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": text_prompt},
        ]

        for image_name in image_names[:MAX_EXPERT_IMAGES]:
            image_path = find_image_path(image_name)
            if image_path is None:
                continue
            content.append({"type": "text", "text": f"图片名称：{image_name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path_to_data_url(image_path),
                    },
                }
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    async def _fallback_expert_reply(
        self,
        history: List[ChatMessage],
        intent: IntentResult,
    ) -> str:
        system_prompt = self.prompt_template or "你是产品手册专家助手。"
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
        extra_user_context = (
            f"路由结果：电商客服\n"
            f"路由原因：{intent.reason}\n\n"
            "请结合完整历史会话，参考以下客服回复示例的语气、结构和处理原则，"
            "给出简洁、礼貌、可执行的客服回复。不要机械复述示例，需根据用户当前问题作答。\n\n"
            "参考示例：\n"
            "1. 用户：请问你们的商品能送到乡镇吗？需要额外加运费吗？多久能到？\n"
            "   客服：您好，我们的商品支持送到大部分乡镇哦，具体能否送达，取决于您的收货地址，"
            "您可以告诉我详细的收货地址，我帮您查询。送到乡镇一般不需要额外加运费，和市区运费一致；"
            "物流时效会比市区稍慢，正常情况下，下单后48小时发货，乡镇地区3-5天可收到，"
            "偏远乡镇可能需要5-7天哦。\n"
            "2. 用户：物流一直显示待揽收，是什么原因？\n"
            "   客服：您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，"
            "一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，"
            "我们会催促快递方尽快上门。\n"
            "3. 用户：我购买的商品，售后维修后，使用不到10天又出现同样的故障，"
            "而且维修人员说这次故障是上次维修不彻底导致的，请问该怎么处理？\n"
            "   客服：您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，"
            "且是上次维修不彻底导致的，属于我们的维修失误，支持免费重新维修，并延长维修质保期。"
            "请您提供维修单号、商品故障描述，我们立即安排专业维修人员处理。"
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
