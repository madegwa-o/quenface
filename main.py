from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, pathlib

import config
from models import Question
from rag import rag_chain, stream_chain

app = FastAPI(title="Qwen RAG API")

# -----------------------------
# CORS CONFIGURATION (Allow All Origins)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


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


@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    dest = pathlib.Path(config.DOCS_DIR) / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Re-index: add doc to existing db
    from rag import embeddings, db
    from langchain_community.document_loaders import TextLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = (
        Docx2txtLoader(str(dest))
        if dest.suffix == ".docx"
        else TextLoader(str(dest))
    )

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    db.add_documents(splitter.split_documents(docs))
    db.persist()

    return {"status": "indexed", "file": file.filename}