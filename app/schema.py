from datetime import date
from pydantic import BaseModel


class ChatStruct(BaseModel):
    query : str
    db_name : str | None = None
    collection_name : str | None = None

class ChatResponse(BaseModel):
    id : int | None = None
    message : str
    reference : str | None = None
    time : float

class DocResponse(BaseModel):
    id : int | None = None
    doc_id : str
    time : float

class CreateDB(BaseModel):
    db_name : str
    collection_name : str

class PredictionResponse(BaseModel):
    prediction : str
    time : float