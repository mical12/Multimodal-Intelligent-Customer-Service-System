from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI()

# ===== 内存存储（后续可以换成数据库）=====
sessions: Dict[str, List[Dict[str, str]]] = {}

# ===== 请求格式 =====
class ChatRequest(BaseModel):
    user_id: str
    message: str

# ===== 简单对话模型（先用假回复）=====
def simple_bot_reply(history: List[Dict[str, str]]) -> str:
    last_user_msg = history[-1]["content"]
    return f"你刚刚说的是：{last_user_msg}"

# ===== 核心接口 =====
@app.post("/chat")
def chat(req: ChatRequest):
    user_id = req.user_id

    # 1. 初始化用户会话
    if user_id not in sessions:
        sessions[user_id] = []

    # 2. 添加用户消息
    sessions[user_id].append({
        "role": "user",
        "content": req.message
    })

    # 3. 生成回复（这里先简单模拟）
    reply = simple_bot_reply(sessions[user_id])

    # 4. 添加助手回复
    sessions[user_id].append({
        "role": "assistant",
        "content": reply
    })

    return {
        "reply": reply,
        "history": sessions[user_id]
    }

if __name__ =='__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )