import os
from operator import itemgetter

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import config


# -----------------------------
# Embeddings
# -----------------------------
embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBED)


# -----------------------------
# Load or Create DB
# -----------------------------

def load_documents():

    docs = []
    docs_path = Path(config.DOCS_DIR)

    for file in docs_path.rglob("*"):

        if file.suffix == ".txt":
            loader = TextLoader(str(file))
            docs.extend(loader.load())

        elif file.suffix == ".docx":
            loader = Docx2txtLoader(str(file))
            docs.extend(loader.load())

    return docs


def load_db():

    if os.path.exists(config.DB_DIR):
        return Chroma(
            persist_directory=config.DB_DIR,
            embedding_function=embeddings
        )

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    split_docs = splitter.split_documents(docs)

    db = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=config.DB_DIR
    )

    db.persist()

    return db


db = load_db()


# -----------------------------
# Retriever
# -----------------------------
retriever = db.as_retriever(
    search_kwargs={"k": config.TOP_K}
)


# -----------------------------
# LLM
# -----------------------------
llm = OllamaLLM(
    model=config.OLLAMA_LLM,
    temperature=config.TEMPERATURE
)


# -----------------------------
# Anti-hallucination prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a STRICT factual assistant.

RULES:
- Answer ONLY using the provided context
- If answer is not in context, say EXACTLY:
  "I don't know based on the provided information."
- Do NOT guess
- Do NOT add outside knowledge
- Be concise and factual

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")


# -----------------------------
# Format docs with citations
# -----------------------------
def format_docs_with_sources(docs):

    formatted = []
    sources = []

    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        sources.append(source)

        formatted.append(
            f"[Source {i+1}: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted), sources


def prepare_context(question):

    docs = retriever.invoke(question)

    context, sources = format_docs_with_sources(docs)

    return {
        "context": context,
        "question": question,
        "sources": sources
    }


# -----------------------------
# Full chain
# -----------------------------
rag_chain = (
    RunnableLambda(prepare_context)
    | {
        "answer": (
            {
                "context": itemgetter("context"),
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | StrOutputParser()
        ),
        "sources": itemgetter("sources"),
    }
)


# -----------------------------
# Streaming chain
# -----------------------------
stream_chain = (
    RunnableLambda(prepare_context)
    | {
        "context": itemgetter("context"),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)