import logging
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from google.genai import types
from core.config import MODEL_NAME
from core.gemini_client import get_gemini_client, generate_content
from utils.prompts import clean_json_output

# Import routers
from routers import github_router, youtube_router

app = FastAPI(
    title="GitHub & YouTube Analysis API",
    description="An API to analyze GitHub repositories and YouTube presentation skills using Gemini AI."
)

# Include routers
app.include_router(github_router.router)
app.include_router(youtube_router.router)


# --- Internal Models ---

class TestPrompt(BaseModel):
    """Internal model for simple Gemini testing."""
    prompt: str


# --- Generic Endpoints ---

@app.get("/health", tags=["Generic"])
async def health():
    """Returns the health status of the API."""
    return {"status": "ok"}


@app.post("/test-gemini", tags=["Generic"])
async def test_gemini(body: TestPrompt):
    """Simple endpoint to test Gemini connectivity with a custom prompt."""
    try:
        client = get_gemini_client()
        parts = [types.Part.from_text(text=body.prompt)]
        contents = [types.Content(role="user", parts=parts)]
        text, req_toks, resp_toks, total_toks = await generate_content(client, MODEL_NAME, contents)

        cleaned = clean_json_output(text)
        return {
            "prompt": body.prompt,
            "raw_response": text,
            "cleaned_response": cleaned,
            "request_tokens": req_toks,
            "response_tokens": resp_toks,
            "total_tokens": total_toks,
        }
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Generic"])
async def root():
    """Root endpoint providing basic API info."""
    return {
        "message": "Welcome to the Analysis API",
        "endpoints": {
            "github": "/github",
            "youtube": "/youtube",
            "health": "/health",
            "docs": "/docs"
        }
    }
