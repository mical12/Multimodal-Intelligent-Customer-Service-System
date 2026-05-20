import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from model import ChatMessage, generate_reply, intent_agent
from rag import get_default_rag
from util import MANUAL_DIR


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
        await intent_agent.warmup_global_rag()
        return

    await asyncio.to_thread(
        get_default_rag().retrieve,
        "如何安装电池？",
        manual_path,
        "空调",
        1,
        0,
    )
    await intent_agent.warmup_global_rag()


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id

    if user_id not in sessions:
        sessions[user_id] = []

    sessions[user_id].append({
        "role": "user",
        "content": req.message,
    })

    reply = await generate_reply(sessions[user_id], user_id=user_id)

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
