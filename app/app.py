# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import time
import random
import json
import os
from schema import ChatResponse, ChatStruct, DocResponse, CreateDB, PredictionResponse
from utils import stream_response, create_dir
from create_db import create_and_save_db
from read_db import SemanticBM25Retriever
from tools.rag_tool.rag_workflow import RAGWorkflow
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
load_dotenv()

create_dir()

llm = Ollama(model="phi3:mini", request_timeout=60.0)
# llm = Gemini(model="models/gemini-1.5-flash-latest", api_key=os.getenv("GOOGLE_API_KEY"))
rag_llm = RAGWorkflow(timeout=120)

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


@app.post("/chat-doc")
async def chat_sample(chat: ChatStruct) -> ChatResponse:

    start = time.time()

    index = SemanticBM25Retriever(collection_name=chat.doc_reference)

    response = await rag_llm.run(query=chat.query, index=index)

    inf_time = float(time.time() - start)

    return ChatResponse(message=str(response), time=inf_time)


@app.post("/chat-sample")
async def chat_sample(chat: ChatStruct) -> ChatResponse:

    start = time.time()

    response = llm.complete(chat.query)

    inf_time = float(time.time() - start)

    return ChatResponse(message=str(response), time=inf_time)


@app.post("/chat-sample-stream")
async def chat_sample_stream(chat: ChatStruct) -> EventSourceResponse:

    return EventSourceResponse(stream_response(llm, chat.query))


@app.post("/upload-single-doc")
async def upload_docs(file: UploadFile) -> DocResponse:

    start = time.time()

    file_content = await file.read()
    file_location = f'./data/{file.filename}'

    with open(file_location, "wb") as f: 
        f.write(file_content)

    inf_time = float(time.time() - start)

    return DocResponse(doc_id=str(file.filename), time=inf_time)


@app.post("/create-db")
async def create_database(db_info : CreateDB) -> DocResponse:

    start = time.time()
    data_dir = "./data"
    save_dir = "./db"

    create_and_save_db(
        data_dir=data_dir,
        save_dir=save_dir,
        db_name=db_info.db_name,
        collection_name=db_info.collection_name)

    inf_time = float(time.time() - start)

    return DocResponse(doc_id=db_info.collection_name, time=inf_time)


@app.post("/emotion-rec")
async def emotion_recognition(file: UploadFile) -> PredictionResponse:

    start = time.time()

    file_content = await file.read()
    file_location = f'./img/{file.filename}'

    # with open(file_location, "wb") as f: 
    #     f.write(file_content)

    inf_time = float(time.time() - start)

    return PredictionResponse(prediction="Concentrating", time=inf_time)