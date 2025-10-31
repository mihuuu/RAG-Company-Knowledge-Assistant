from __future__ import annotations
import os, glob, uuid, asyncio, traceback
from typing import Iterable, List, Dict, Any
from pathlib import Path

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader,TextLoader

from .utils import get_vector_store
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore

DATA_DIR = os.getenv("DATA_DIR", "data")

def load_docs(base: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []

    # recurse through all files under base
    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue
        ext = os.path.splitext(path)[1].lower()

        relative_path = os.path.relpath(path, base)
        category = relative_path.split(os.sep)[0] if os.sep in relative_path else "general"

        try:
            cur_docs = []
            if ext == ".md":
                loader = UnstructuredMarkdownLoader(path)
                cur_docs.extend(loader.load())
            elif ext  == ".pdf":
                loader = PyMuPDFLoader(path)
                cur_docs.extend(loader.load())
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(path)
                cur_docs.extend(loader.load())
            elif ext in [".txt"]:
                loader = TextLoader(path, encoding="utf8")
                cur_docs.extend(loader.load())
            else:
                print(f"INGEST WARNING: unsupported file type {path}")
            
            # Add metadata
            for d in cur_docs:
                d.metadata["category"] = category
                docs.append(d)

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

async def create_index(store: AsyncPGVectorStore):
    index = HNSWIndex(
        name="hnsw_idx",
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        m=16,
        ef_construction=64
    )
    await store.aapply_vector_index(index, concurrently=True)
    print("Index created successfully")


async def run_ingest_async() -> dict:
   docs = load_docs()
   chunks = split_chunks(docs)
   store = await get_vector_store()
   await store.aadd_documents(chunks)
   print(f"INGEST: {len(docs)} docs, {len(chunks)} chunks")

   await create_index(store)

   return {"documents": len(docs), "chunks": len(chunks)}


