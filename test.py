import requests

resp = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"user_id": "u1", "message": "你好"}
)

print(resp.json())