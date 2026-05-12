import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from model import ChatMessage, generate_reply
from rag import get_default_rag
from util import MANUAL_DIR


MAX_REPLY_RETRIES = 2
FALLBACK_REPLY = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await warmup_rag()
    yield


app = FastAPI(lifespan=lifespan)

# In-memory storage, can be replaced by a database later.
sessions: Dict[str, List[ChatMessage]] = {}


async def warmup_rag():
    manual_path = MANUAL_DIR / "空调手册.txt"
    if not manual_path.exists():
        return

    await asyncio.to_thread(
        get_default_rag().retrieve,
        "如何安装电池？",
        manual_path,
        "空调",
        1,
        0,
    )


class ChatRequest(BaseModel):
    user_id: str
    message: str


async def generate_reply_with_retry(
    history: List[ChatMessage],
    user_id: str,
    max_retries: int = MAX_REPLY_RETRIES,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await generate_reply(history, user_id=user_id)
        except Exception as exc:
            last_error = exc
            print(
                f"user_id={user_id} reply attempt={attempt}/{max_retries} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1)

    print(
        f"user_id={user_id} all reply attempts failed: "
        f"{type(last_error).__name__ if last_error else 'UnknownError'}: {last_error}"
    )
    return FALLBACK_REPLY


@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id

    if user_id not in sessions:
        sessions[user_id] = []

    sessions[user_id].append({
        "role": "user",
        "content": req.message,
    })

    reply = await generate_reply_with_retry(sessions[user_id], user_id=user_id)

    sessions[user_id].append({
        "role": "assistant",
        "content": reply,
    })

    return {
        "reply": reply,
        "history": sessions[user_id],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
