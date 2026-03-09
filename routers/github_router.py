import logging
import traceback
import json
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from google.genai import types
from core.config import MODEL_NAME
from core.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    SemanticAnalyzeResponse
)
from utils.repo_utils import clone_repo, get_code_files
from core.gemini_client import get_gemini_client, generate_content
from utils.embed_utils import embed_files
from utils.prompts import build_prompt, clean_json_output
from utils.doc_extract_utils import extract_github_url_from_document

router = APIRouter(prefix="/github", tags=["GitHub Repository"])

# --- Helper Functions (GitHub Related) ---

async def run_analysis(repo_url: str, screenshots: Optional[List[UploadFile]] = None) -> Dict[str, Any]:
    """
    Clones a repository, gathers code, and performs a direct analysis using Gemini.
    Optionally accepts screenshots to enhance the analysis.
    """
    clone_dir = await clone_repo(repo_url)
    code_files = await get_code_files(clone_dir)

    file_summaries = "\n\n".join([
        f"\n\nFile: {f['file']}\n---\n{f['content']}" for f in code_files
    ])

    prompt = build_prompt(repo_url, file_summaries, mode="direct")
    client = get_gemini_client()

    parts = [types.Part.from_text(text=prompt)]
    
    # Process optional screenshots
    if screenshots:
        for i, img in enumerate(screenshots):
            img_bytes = await img.read()
            filename = img.filename.lower() if img.filename else ""
            mime_type = "image/jpeg"
            if filename.endswith(".png"):
                mime_type = "image/png"
            elif filename.endswith(".webp"):
                mime_type = "image/webp"
            
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

    contents = [types.Content(role="user", parts=parts)]
    text, req_toks, resp_toks, total_toks = await generate_content(client, MODEL_NAME, contents)

    try:
        cleaned = clean_json_output(text)
        parsed_response = json.loads(cleaned)
    except Exception:
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


async def run_semantic_analysis(repo_url: str, query: str, screenshots: Optional[List[UploadFile]] = None) -> Dict[str, Any]:
    """
    Performs a RAG-based analysis of the repository code using a semantic query.
    """
    clone_dir = await clone_repo(repo_url)
    code_files = await get_code_files(clone_dir)

    if code_files:
        vectorstore = embed_files(code_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])
    else:
        context = ""

    prompt = build_prompt(repo_url, context, mode="semantic")
    client = get_gemini_client()

    parts = [types.Part.from_text(text=prompt)]
    contents = [types.Content(role="user", parts=parts)]
    text, req_toks, resp_toks, total_toks = await generate_content(client, MODEL_NAME, contents)

    try:
        cleaned = clean_json_output(text)
        parsed_response = json.loads(cleaned)
    except Exception:
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


# --- Endpoints ---

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(payload: AnalyzeRequest):
    """
    Standard analysis of a GitHub repository via direct URL.
    Scans repository files and provides an expert review score.
    """
    try:
        return await run_analysis(payload.repo_url)
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic-analyze", response_model=SemanticAnalyzeResponse)
async def semantic_analyze(payload: AnalyzeRequest, query: str = Query("Get All Files and their Code")):
    """
    Semantic analysis of a GitHub repository based on a natural language query.
    Uses RAG to find relevant code snippets for the analysis.
    """
    try:
        return await run_semantic_analysis(payload.repo_url, query=query)
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-doc", response_model=AnalyzeResponse)
async def analyze_document(
    document: UploadFile = File(...),
    images: Optional[List[UploadFile]] = File(default=None)
):
    """
    Extracts a GitHub URL from a PDF/DOCX file and runs repository analysis.
    Accepts up to 4 optional images (screenshots) to enhance the expert review.
    """
    try:
        if images and len(images) > 4:
            images = images[:4]
            logging.info("Truncated images to 4 for document analysis.")

        file_content = await document.read()
        filename = document.filename or "unknown.pdf"
        
        try:
            repo_url = extract_github_url_from_document(file_content, filename)
            logging.info(f"Extracted GitHub URL from document: {repo_url}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return await run_analysis(repo_url, screenshots=images)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
