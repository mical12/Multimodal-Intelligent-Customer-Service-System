from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from model import ChatMessage, generate_reply


app = FastAPI()

# In-memory storage, can be replaced by a database later.
sessions: Dict[str, List[ChatMessage]] = {}


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
