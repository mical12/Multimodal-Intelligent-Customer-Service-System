import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from model import ChatMessage, generate_reply, intent_agent
from rag import MANUAL_DIR as RAG_MANUAL_DIR
from rag import get_default_rag, warmup_targets


@asynccontextmanager
async def lifespan(app: FastAPI):
    await warmup_rag()
    yield


app = FastAPI(lifespan=lifespan)

# In-memory storage, can be replaced by a database later.
sessions: Dict[str, List[ChatMessage]] = {}


async def warmup_rag():
    rag = get_default_rag()

    print("warmup: building all manual chunk/vector caches...")
    warmed = await asyncio.to_thread(rag.warmup_all, RAG_MANUAL_DIR)
    print(f"warmup: manual chunk/vector caches ready, targets={len(warmed)}")

    print("warmup: building bm25 caches...")
    await asyncio.to_thread(warmup_bm25_indexes)
    print("warmup: bm25 caches ready")

    print("warmup: loading reranker once...")
    await asyncio.to_thread(warmup_reranker_once)
    print("warmup: reranker ready")

    print("warmup: building global rag routing indexes...")
    await intent_agent.warmup_global_rag()
    print("warmup: global rag routing indexes ready")


def warmup_bm25_indexes():
    rag = get_default_rag()
    for manual_path, product_name in warmup_targets(RAG_MANUAL_DIR):
        rag.retrieve_bm25(
            question="warmup",
            manual_path=manual_path,
            product_name=product_name,
            top_k=1,
        )


def warmup_reranker_once():
    targets = warmup_targets(RAG_MANUAL_DIR)
    if not targets:
        return
    manual_path, product_name = targets[0]
    get_default_rag().retrieve_with_bm25_rerank(
        question="warmup",
        manual_path=manual_path,
        product_name=product_name,
        embedding_top_k=1,
        bm25_top_k=1,
        final_top_k=1,
        neighbor_count=0,
    )


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
