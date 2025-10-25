from __future__ import annotations
import os, glob, uuid, asyncio, traceback
from typing import Iterable, List, Dict, Any
from pathlib import Path

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader,TextLoader

from .utils import get_vector_store
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy

DATA_DIR = os.getenv("DATA_DIR", "data")

def load_docs(base: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []

    # recurse through all files under base
    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".md":
                loader = UnstructuredMarkdownLoader(path)
                docs.extend(loader.load())
            elif ext  == ".pdf":
                loader = PyMuPDFLoader(path)
                docs.extend(loader.load())
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(path)
                docs.extend(loader.load())
            elif ext in [".txt"]:
                loader = TextLoader(path, encoding="utf8")
                docs.extend(loader.load())
            else:
                print(f"INGEST WARNING: unsupported file type {path}")
        except Exception:
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()

    return docs
        

def split_chunks(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120
    )
    try:
        return splitter.split_documents(docs)
    except Exception:
        print(f"INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise


async def run_ingest_async() -> dict:
   docs = load_docs()
   chunks = split_chunks(docs)
   store = await get_vector_store()
   await store.aadd_documents(chunks)
   print(f"INGEST: {len(docs)} docs, {len(chunks)} chunks")
   return {"documents": len(docs), "chunks": len(chunks)}


