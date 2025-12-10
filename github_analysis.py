import os
import asyncio
import logging
import traceback
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from git import Repo
import aiofiles
import uvicorn

from google import genai
from google.genai import types

# ----------------------------
# Configs
# ----------------------------
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
    analysis_text: str

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
    file_label = "File List" if mode == "direct" else "File Context"
    return f"""
You are a senior software architect.

Analyze the GitHub repository at {repo_url}.

Section A — Repository Overview:
- High-Level Summary
- Tech Stack
- Architecture
- Code Quality (/10)
- Complexity Areas
- Security Issues
- Suggestions
- Readme Review
- Testing
- Deployment Details
- Code Evaluation (/10) with Deductions:
  - Score: X/10
  - Rationale: 2–4 sentences summarizing the score reasoning.
  - Deductions: list items where each item is "(-N) Reason for deduction"

Section B — File-by-File Analysis:
Use the {file_label} below. For each file mentioned, provide:
- 1–3 sentence summary of its responsibilities
- Mention problems (security, performance, docs/tests missing)
- If not readable, say: "unable to access contents"

{file_label}:
{context}
"""


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

    return {
        "repo_url": repo_url,
        "request_tokens": req_toks,
        "response_tokens": resp_toks,
        "total_tokens": total_toks,
        "model": MODEL_NAME,
        "analysis_text": text,
    }

# API Routes
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(payload: AnalyzeRequest):
    try:
        return await run_analysis(payload.repo_url)
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
