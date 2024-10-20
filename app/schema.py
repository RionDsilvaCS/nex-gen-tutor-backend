from datetime import date
from pydantic import BaseModel


class ChatStruct(BaseModel):
    query : str
    doc_reference : str | None = None

class ChatResponse(BaseModel):
    id : int | None = None
    message : str
    reference : str | None = None
    time : float