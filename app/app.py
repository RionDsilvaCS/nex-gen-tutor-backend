from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
import time
from schema import ChatResponse, ChatStruct
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from sse_starlette import EventSourceResponse
import json
from utils import stream_response

llm = Ollama(model="phi3:mini", request_timeout=60.0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"message": "Bazinga 🎉"}


@app.post("/chat-sample")
async def chat_sample(chat: ChatStruct) -> ChatResponse:

    start = time.time()

    response = llm.complete(chat.query)

    inf_time = float(time.time() - start)

    return ChatResponse(message=str(response), time=inf_time)


@app.post("/chat-sample-stream")
async def chat_sample_stream(chat: ChatStruct) -> EventSourceResponse:

    return EventSourceResponse(stream_response(llm, chat.query))
