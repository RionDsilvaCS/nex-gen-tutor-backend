from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
import time
from schema import ChatResponse, ChatStruct

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    return {"message": "Bazinga 🎉"}

@app.post("/chat-sample")
async def chat_sample(chat: ChatStruct):

    start = time.time()

    sample_messages = [
    "Hello there! 👋 I'm your friendly AI assistant, here to help you with anything you need. How can I assist you today?",
    "Welcome! I'm your AI chatbot, ready to provide answers and support. What can I help you with right now?",
    "Hi! 🌟 I'm here to make your experience enjoyable and informative. What questions do you have for me today?",
    "Greetings! I'm your AI companion, dedicated to helping you find the information you seek. Let's get started!",
    "Hey there! 🎉 I'm excited to help you out! Whether it's questions or guidance, just let me know how I can assist!"]

    idx = random.randint(1, 5)

    # time.sleep(2)

    inf_time = float(time.time() - start)

    return ChatResponse(message=sample_messages[idx-1], time=inf_time)
