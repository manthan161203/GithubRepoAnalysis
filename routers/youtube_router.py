import logging
import traceback
import json
import asyncio
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File
from google.genai import types
from google.genai import errors as genai_errors
from core.config import MODEL_NAME
from core.schemas import (
    YoutubeAnalyzeRequest,
    YoutubeAnalyzeResponse
)
from core.gemini_client import get_gemini_client, generate_content, generate_content_youtube
from utils.prompts import build_youtube_prompt, clean_json_output
from urllib.parse import urlparse, parse_qs

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

router = APIRouter(prefix="/youtube", tags=["YouTube Analysis"])

# --- Helper Functions (YouTube Related) ---

async def run_youtube_analysis(youtube_urls: List[str]) -> Dict[str, Any]:
    """Analyze one or more YouTube videos using Gemini's native YouTube support."""
    client = get_gemini_client()
    prompt = build_youtube_prompt()
    try:
        text, req_toks, resp_toks, total_toks = await generate_content_youtube(client, MODEL_NAME, youtube_urls, prompt)
    except Exception as e:
        is_permission_error = False
        try:
            if isinstance(e, genai_errors.ClientError):
                if getattr(e, 'status_code', None) == 403 or 'PERMISSION_DENIED' in str(e):
                    is_permission_error = True
        except Exception:
            is_permission_error = False

        if is_permission_error:
            logging.warning("Native YouTube analysis permission denied; falling back to transcript-based analysis.")
            return await run_youtube_transcript_analysis(youtube_urls)
        raise

    try:
        cleaned = clean_json_output(text)
        parsed_response = json.loads(cleaned)
    except Exception:
        logging.error("Invalid JSON from model for YouTube analysis, returning raw text.")
        parsed_response = {"raw_output": text}

    return {
        "youtube_urls": youtube_urls,
        "model": MODEL_NAME,
        "request_tokens": req_toks,
        "response_tokens": resp_toks,
        "total_tokens": total_toks,
        "analysis": parsed_response,
    }


def _extract_video_id(youtube_url: str) -> Optional[str]:
    """Extracts the YouTube video ID from various URL formats."""
    try:
        parsed = urlparse(youtube_url)
        hostname = (parsed.hostname or "").lower()
        if "youtu.be" in hostname:
            return parsed.path.lstrip("/")
        if "youtube" in hostname:
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
            parts = [p for p in parsed.path.split("/") if p]
            if parts:
                return parts[-1]
    except Exception:
        return None
    return None


async def _fetch_transcript_for_video(youtube_url: str) -> str:
    """Fetches the transcript for a private/public YouTube video using secondary API."""
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube_transcript_api package is not installed.")

    video_id = _extract_video_id(youtube_url)
    if not video_id:
        raise ValueError(f"Could not extract video id from URL: {youtube_url}")

    try:
        transcript_list = await asyncio.to_thread(YouTubeTranscriptApi().fetch, video_id)
        transcript_texts = []
        for seg in transcript_list:
            if isinstance(seg, dict):
                transcript_texts.append(seg.get("text", ""))
            else:
                text_val = getattr(seg, "text", None) or getattr(seg, "content", None) or str(seg)
                transcript_texts.append(text_val)

        full_transcript = "\n".join(transcript_texts)
        logging.info("Full transcript for %s (video_id=%s) fetched.", youtube_url, video_id)
        return full_transcript
    except Exception as e:
        logging.error("Transcript fetch failed for %s: %s", youtube_url, traceback.format_exc())
        raise


async def run_youtube_transcript_analysis(youtube_urls: List[str]) -> Dict[str, Any]:
    """Fetch transcripts for YouTube URLs and send to Gemini for rubric analysis."""
    transcripts = []
    for u in youtube_urls:
        t = await _fetch_transcript_for_video(u)
        transcripts.append(f"--- Transcript for {u} ---\n" + t)

    combined = "\n\n".join(transcripts)
    prompt = build_youtube_prompt() + "\n\nTranscript (truncated to 15000 chars):\n" + combined[:15000]

    client = get_gemini_client()
    parts = [types.Part.from_text(text=prompt)]
    contents = [types.Content(role="user", parts=parts)]

    text, req_toks, resp_toks, total_toks = await generate_content(client, MODEL_NAME, contents)

    try:
        cleaned = clean_json_output(text)
        parsed_response = json.loads(cleaned)
    except Exception:
        logging.error("Invalid JSON from model for transcript analysis, returning raw text.")
        parsed_response = {"raw_output": text}

    return {
        "youtube_urls": youtube_urls,
        "model": MODEL_NAME,
        "request_tokens": req_toks,
        "response_tokens": resp_toks,
        "total_tokens": total_toks,
        "analysis": parsed_response,
    }


# --- Endpoints ---

@router.post("/analyze", response_model=YoutubeAnalyzeResponse)
async def analyze_youtube(body: YoutubeAnalyzeRequest):
    """
    Analyze presentation skills from YouTube video(s) using native Gemini support.
    Evaluates body language, facial expression, tonality, structure, and impact.
    """
    try:
        for u in body.youtube_urls:
            if not (u.startswith("https://www.youtube.com/") or u.startswith("https://youtu.be/")):
                raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
        return await run_youtube_analysis(body.youtube_urls)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-transcript", response_model=YoutubeAnalyzeResponse)
async def analyze_youtube_transcript(body: YoutubeAnalyzeRequest):
    """
    Analyze presentation skills from YouTube video(s) using fetched transcripts.
    Fallback or alternative to native YouTube support.
    """
    try:
        for u in body.youtube_urls:
            if not (u.startswith("https://www.youtube.com/") or u.startswith("https://youtu.be/")):
                raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
        return await run_youtube_transcript_analysis(body.youtube_urls)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
