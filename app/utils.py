import json
from llama_index.core.llms import ChatMessage
import os

def create_dir():
    directories = ["./data", "./db", "./img"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

async def stream_response(llm, query: str):
    messages = [
        ChatMessage(
            role="system", content="You are a young fun public speaker"
        ),
        ChatMessage(role="user", content=query),
    ]
    try:
        resp = llm.stream_chat(messages)
        for r in resp:
            yield {
                    "data": json.dumps(f"{r.delta}"),
                    "event": "data",
            }
        yield {"event":"end"}
    except:
        yield {
            "event": "error",
            "data": json.dumps(
                {"status_code": 500, "message": "Internal Server Error"}
            ),
        }
        raise