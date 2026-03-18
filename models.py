from pydantic import BaseModel

class Question(BaseModel):
    query: str

class KnowledgeText(BaseModel):
    text: str
    category: str = "general"