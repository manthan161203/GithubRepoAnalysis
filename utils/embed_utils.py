from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import CHUNK_SIZE, CHUNK_OVERLAP


def embed_files(code_files: List[Dict[str, str]]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs, metadatas = [], []
    for f in code_files:
        for chunk in splitter.split_text(f["content"]):
            docs.append(chunk)
            metadatas.append({"source": f["file"]})
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    return FAISS.from_texts(docs, embeddings, metadatas=metadatas)
