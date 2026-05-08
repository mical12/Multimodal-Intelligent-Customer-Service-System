import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from openai import AsyncOpenAI


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
MANUAL_ALIAS_PATH = BASE_DIR / "manual_aliases.yaml"
MANUAL_SUMMARY_PATH = BASE_DIR / "manual_summaries.yaml"
MANUAL_DIR = BASE_DIR / "手册"
IMAGE_DIR = MANUAL_DIR / "插图"
INTENT_LOG_PATH = BASE_DIR / "intent_log.jsonl"
FIXED_TEST_REPLY = "您好，您的问题已收到，我们会尽快为您处理。"
LLM_ROUTER_PRODUCTS = {"相机", "专业相机"}

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
    reason: str = ""


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

    return {
        str(product_name): [str(alias) for alias in alias_list if str(alias).strip()]
        for product_name, alias_list in aliases.items()
        if isinstance(alias_list, list)
    }


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
        "reason": intent.reason,
    }

    with INTENT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_item, ensure_ascii=False) + "\n")


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
        self.manual_summaries = load_manual_summaries()

    async def recognize(self, history: List[ChatMessage]) -> IntentResult:
        question = latest_user_message(history)
        manual_path = self._match_manual(question)

        if manual_path is not None:
            product_name = self._product_name_from_manual(manual_path)
            if product_name not in LLM_ROUTER_PRODUCTS:
                return IntentResult(
                    agent_type="expert",
                    product_name=product_name,
                    manual_path=manual_path,
                    reason="规则命中产品手册",
                )

        llm_manual_path = await self._match_manual_with_llm(question, manual_path)
        if llm_manual_path is not None:
            product_name = self._product_name_from_manual(llm_manual_path)
            return IntentResult(
                agent_type="expert",
                product_name=product_name,
                manual_path=llm_manual_path,
                reason="大模型结合手册摘要命中产品手册",
            )

        return IntentResult(
            agent_type="customer",
            reason="规则和大模型均未命中产品手册，走通用客服",
        )

    def _match_manual(self, question: str) -> Path | None:
        normalized_question = normalize_match_text(question)
        best_match: tuple[int, Path] | None = None

        for manual_path in self._manual_files():
            product_name = self._product_name_from_manual(manual_path)
            aliases = [product_name, *self.manual_aliases.get(product_name, [])]

            for alias in aliases:
                normalized_alias = normalize_match_text(alias)
                if not normalized_alias or normalized_alias not in normalized_question:
                    continue
                alias_length = len(normalized_alias)
                if best_match is None or alias_length > best_match[0]:
                    best_match = (alias_length, manual_path)

        return best_match[1] if best_match else None

    async def _match_manual_with_llm(
        self,
        question: str,
        rule_manual_path: Path | None = None,
    ) -> Path | None:
        manual_options = self._manual_options_for_llm(rule_manual_path)
        if not manual_options:
            return None

        system_prompt = (
            "你是电商客服系统的产品手册路由器。"
            "请根据用户问题和每本手册摘要，判断问题是否应该交给某一本产品手册处理。"
            "如果问题是售后、物流、发票、退换货、投诉等通用客服问题，返回 customer。"
            "只输出 JSON，不要输出解释。"
        )
        user_prompt = (
            f"用户问题：{question}\n\n"
            "可选产品手册摘要：\n"
            f"{manual_options}\n\n"
            '输出格式：{"agent_type":"expert或customer","product_name":"产品名或null"}'
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

        product_name = str(result.get("product_name") or "").strip()
        if not product_name:
            return None

        return self._manual_path_by_product_name(product_name)

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
            if path.name != "汇总英文手册.txt"
        ]

    @staticmethod
    def _product_name_from_manual(manual_path: Path) -> str:
        return manual_path.stem.removesuffix("手册")


class ExpertAgent(BaseAgent):
    """Handle product manual questions. RAG and image retrieval will be added here later."""

    async def reply(self, history: List[ChatMessage], intent: IntentResult) -> str:
        product_name = intent.product_name or "相关产品"
        return (
            f"已识别为【{product_name}】相关问题，后续这里会接入专家 agent："
            "先检索对应手册，再结合历史会话和需要匹配插图。"
        )


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
    # if intent.agent_type == "expert":
    #    return await expert_agent.reply(history, intent)

    # return await customer_agent.reply(history, intent)

    return FIXED_TEST_REPLY
