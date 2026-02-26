from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from models import Question
from rag import rag_chain, stream_chain

app = FastAPI(title="Qwen RAG API")


@app.get("/")
def root():
    return {"status": "running"}


# -----------------------------
# Standard answer with citations
# -----------------------------
@app.post("/ask")
def ask(question: Question):

    result = rag_chain.invoke(question.query)

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


# -----------------------------
# Streaming endpoint
# -----------------------------
@app.post("/ask-stream")
def ask_stream(question: Question):

    def generate():

        for chunk in stream_chain.stream(question.query):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )