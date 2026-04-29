import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from openai import AsyncOpenAI


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
MANUAL_DIR = BASE_DIR / "手册"
IMAGE_DIR = MANUAL_DIR / "插图"

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

    async def recognize(self, history: List[ChatMessage]) -> IntentResult:
        question = latest_user_message(history)
        manual_path = self._match_manual(question)

        if manual_path is not None:
            product_name = self._product_name_from_manual(manual_path)
            return IntentResult(
                agent_type="expert",
                product_name=product_name,
                manual_path=manual_path,
                reason="命中产品手册",
            )

        return IntentResult(
            agent_type="customer",
            reason="未命中产品手册，走通用客服",
        )

    def _match_manual(self, question: str) -> Path | None:
        for manual_path in self._manual_files():
            product_name = self._product_name_from_manual(manual_path)
            if product_name and product_name in question:
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
            f"路由结果：通用客服\n"
            f"路由原因：{intent.reason}\n\n"
            "请结合完整历史会话，给出简洁、礼貌、可执行的客服回复。"
        )
        return await self._chat_with_history(history, system_prompt, extra_user_context)


config = load_config()
intent_agent = IntentAgent(get_agent_config(config, "intent"))
expert_agent = ExpertAgent(get_agent_config(config, "expert"))
customer_agent = CustomerAgent(get_agent_config(config, "customer"))


async def generate_reply(history: List[ChatMessage]) -> str:
    intent = await intent_agent.recognize(history)

    if intent.agent_type == "expert":
        return await expert_agent.reply(history, intent)

    return await customer_agent.reply(history, intent)
