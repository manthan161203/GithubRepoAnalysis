import os
import asyncio
import logging
import traceback
import json
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from git import Repo
import aiofiles
import uvicorn

from google import genai
from google.genai import types

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ----------------------------
# Configs
# ----------------------------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MODEL_NAME = "gemini-2.5-flash-lite"


# ----------------------------
# Environment & Logging
# ----------------------------
def setup_environment() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


setup_environment()


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="GitHub Repo Analysis API")


# ----------------------------
# Schemas
# ----------------------------
class AnalyzeRequest(BaseModel):
    repo_url: HttpUrl


class AnalyzeResponse(BaseModel):
    repo_url: HttpUrl
    request_tokens: int
    response_tokens: int
    total_tokens: int
    model: str
    analysis: Dict[str, Any]  # structured JSON response


class SemanticAnalyzeResponse(BaseModel):
    repo_url: HttpUrl
    query: str
    response: Dict[str, Any]  # structured JSON response
    model: str
    request_tokens: int
    response_tokens: int
    total_tokens: int


# ----------------------------
# Repo Management
# ----------------------------
async def clone_repo(repo_url: str, clone_dir: str = "cloned_repo") -> str:
    if os.path.exists(clone_dir):
        import shutil
        shutil.rmtree(clone_dir, ignore_errors=True)
    await asyncio.to_thread(Repo.clone_from, repo_url, clone_dir)
    return clone_dir


async def get_code_files(clone_dir: str, max_chars: int = 5000) -> List[Dict[str, str]]:
    code_files = []
    for root, _, files in os.walk(clone_dir):
        if '.git' in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = await f.read()
                code_files.append({
                    "file": os.path.relpath(file_path, clone_dir),
                    "content": content[:max_chars]
                })
            except Exception as e:
                logging.warning(f"Could not read {file_path}: {e}")
    return code_files


# ----------------------------
# Embedding & Retrieval
# ----------------------------
def embed_files(code_files: List[Dict[str, str]]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs, metadatas = [], []
    for f in code_files:
        for chunk in splitter.split_text(f["content"]):
            docs.append(chunk)
            metadatas.append({"source": f["file"]})
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    return FAISS.from_texts(docs, embeddings, metadatas=metadatas)


# ----------------------------
# Gemini Helpers
# ----------------------------
def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY in environment.")
    return genai.Client(api_key=api_key)


async def count_tokens(client: genai.Client, contents: list) -> int:
    result = await client.aio.models.count_tokens(model=MODEL_NAME, contents=contents)
    return getattr(result, "total_tokens", 0)


async def generate_content(client: genai.Client, contents: list, prompt: str):
    request_tokens = await count_tokens(client, contents)
    response = await client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_modalities=["TEXT"]
        )
    )
    response_text = response.text or ""
    usage = getattr(response, "usage_metadata", None)
    response_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    total_tokens = getattr(usage, "total_token_count", request_tokens)
    return response_text, request_tokens, response_tokens, total_tokens


# ----------------------------
# Prompt Template
# ----------------------------
def build_prompt(repo_url: str, context: str, mode: str = "direct") -> str:
    return f"""
You are an expert software reviewer and UX auditor.

Analyze the GitHub repository: {repo_url}.
Use the provided code context to evaluate.

Return the analysis STRICTLY as a well-formatted JSON object
with the following schema:

{{
  "total_score": <integer 0–100>,
  "section_scores": {{
    "Business Understanding": <integer 0–10>,
    "Objectives Clarity": <integer 0–10>,
    "Design Rationale": <integer 0–10>,
    "Responsiveness": <integer 0–10>,
    "Feature Completeness": <integer 0–20>,
    "Brand Alignment": <integer 0–10>,
    "UX Structure & CTAs": <integer 0–20>,
    "Personal Impression": <integer 0–10>
  }},
  "section_reasoning": {{
    "Business Understanding": <string>,
    "Objectives Clarity": <string>,
    "Design Rationale": <string>,
    "Responsiveness": <string>,
    "Feature Completeness": <string>,
    "Brand Alignment": <string>,
    "UX Structure & CTAs": <string>,
    "Personal Impression": <string>
  }},
  "overall_feedback": <string>
}}

Rules:
- Do not add comments or explanation outside the JSON.
- Output must be valid JSON.
- Base scoring on repository architecture, clarity, design, code quality, and usability.

Repository context (truncated for length):
{context}
"""

# ----------------------------
# Json Parsing
# ----------------------------
def clean_json_output(text: str) -> str:
    """
    Cleans model output:
    - Removes Markdown fences like ``````
    - Strips whitespace
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove opening ```json or ```
        text = text.split("```", 1)[-1]
        if text.lower().startswith("json"):
            text = text[4:].strip()
        # Remove closing ```
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text


# ----------------------------
# Core Workflows
# ----------------------------
async def run_analysis(repo_url: str) -> Dict[str, Any]:
    clone_dir = await clone_repo(repo_url)
    code_files = await get_code_files(clone_dir)
    file_summaries = "\n\n".join([
        f"\n\nFile: {f['file']}\n---\n{f['content']}"
        for f in code_files
    ])

    prompt = build_prompt(repo_url, file_summaries, mode="direct")
    client = get_gemini_client()
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    text, req_toks, resp_toks, total_toks = await generate_content(client, contents, prompt)

    try:
        cleaned = clean_json_output(text)
        parsed_response = json.loads(cleaned)
    except json.JSONDecodeError:
        logging.error("Invalid JSON from model, returning raw text.")
        parsed_response = {"raw_output": text}

    return {
        "repo_url": repo_url,
        "request_tokens": req_toks,
        "response_tokens": resp_toks,
        "total_tokens": total_toks,
        "model": MODEL_NAME,
        "analysis": parsed_response,
    }


async def run_semantic_analysis(repo_url: str, query: str) -> Dict[str, Any]:
    clone_dir = await clone_repo(repo_url)
    code_files = await get_code_files(clone_dir)
    vectorstore = embed_files(code_files)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = build_prompt(repo_url, context, mode="semantic")
    client = get_gemini_client()
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    text, req_toks, resp_toks, total_toks = await generate_content(client, contents, prompt)

    try:
        cleaned = clean_json_output(text)
        parsed_response = json.loads(cleaned)
    except json.JSONDecodeError:
        logging.error("Invalid JSON from model, returning raw text.")
        parsed_response = {"raw_output": text}

    return {
        "repo_url": repo_url,
        "query": query,
        "response": parsed_response,
        "model": MODEL_NAME,
        "request_tokens": req_toks,
        "response_tokens": resp_toks,
        "total_tokens": total_toks,
    }


# ----------------------------
# API Routes
# ----------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(payload: AnalyzeRequest):
    try:
        return await run_analysis(payload.repo_url)
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/semantic-analyze", response_model=SemanticAnalyzeResponse)
async def semantic_analyze(payload: AnalyzeRequest, query: str = Query("Get All Files and their Code")):
    try:
        return await run_semantic_analysis(payload.repo_url, query=query)
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=True)
