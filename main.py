from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, pathlib, os

import config
from models import Question, KnowledgeText
from rag import rag_chain, stream_chain

app = FastAPI(title="Qwen RAG API")

def remove_document_from_chroma(file_path_str: str):
    try:
        from rag import db
        res = db.get(where={"source": file_path_str})
        ids = res.get("ids", []) if res else []
        if ids:
            db.delete(ids=ids)
    except Exception as e:
        print(f"Error removing {file_path_str} from chroma: {e}")

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

@app.get("/test")
def test():
    return {"message": "CORS test successful", "timestamp": "2026-03-02"}

@app.post("/ask")
def ask(question: Question):
    result = rag_chain.invoke(question.query)
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


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
        else TextLoader(str(dest), autodetect_encoding=True)
    )

    # Remove old version if updating
    remove_document_from_chroma(str(dest))

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    db.add_documents(splitter.split_documents(docs))
    # db.persist() # Deprecated in newer ChromaDB versions

    return {"status": "indexed", "file": file.filename}

@app.post("/add-text")
def add_text_knowledge(knowledge: KnowledgeText):
    import time
    filename = f"knowledge_{knowledge.category}_{int(time.time())}.txt"
    dest = pathlib.Path(config.DOCS_DIR) / filename
    
    with dest.open("w", encoding="utf-8") as f:
        f.write(knowledge.text)

    # Re-index
    from rag import embeddings, db
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = TextLoader(str(dest), autodetect_encoding=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    db.add_documents(splitter.split_documents(docs))
    # db.persist() # Deprecated

    return {"status": "indexed", "file": filename}


# Added: List, view, and delete documents
@app.get("/documents")
def list_documents():
    docs_path = pathlib.Path(config.DOCS_DIR)
    
    if not docs_path.exists():
        return {"documents": []}
    
    documents = []
    for file in docs_path.rglob("*"):
        if file.is_file() and file.suffix in [".txt", ".docx"]:
            documents.append({
                "filename": file.name,
                "size": file.stat().st_size,
                "type": str(file.suffix).replace(".", "")
            })
    
    return {"documents": documents}

@app.get("/documents/{filename}")
def get_document(filename: str):
    file_path = pathlib.Path(config.DOCS_DIR) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )

@app.delete("/documents/all")
def delete_all_documents():
    docs_path = pathlib.Path(config.DOCS_DIR)
    if not docs_path.exists():
        return {"status": "cleared", "count": 0}
        
    count: int = 0
    try:
        from rag import db
        try:
            all_ids = db.get().get("ids", [])
            if all_ids:
                db.delete(ids=all_ids)
        except Exception as err:
            print(f"Warning: Failed to clear ChromaDB vectors: {err}")

        for file in docs_path.rglob("*"):
            if file.is_file() and file.suffix in [".txt", ".docx"]:
                try:
                    os.remove(file)
                    count = count + 1
                except Exception:
                    pass
        return {"status": "cleared", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear all: {str(e)}")

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    file_path = pathlib.Path(config.DOCS_DIR) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        os.remove(file_path)
        remove_document_from_chroma(str(file_path))
        return {"status": "deleted", "file": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")
