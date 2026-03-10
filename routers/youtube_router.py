import logging
import traceback
import json
import asyncio
import os
from typing import Any, Dict, Optional, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File
from google.genai import types
from core.config import (
    MODEL_NAME,
    VIDEO_FILE_MAX_WAIT_SECONDS,
    VIDEO_FILE_POLL_INTERVAL_SECONDS,
    YTDLP_COOKIES_FILE,
    YTDLP_COOKIES_FROM_BROWSER,
    DOWNLOADED_VIDEO_DIR,
)
from core.schemas import (
    YoutubeAnalyzeRequest,
    YoutubeAnalyzeResponse
)
from core.gemini_client import (
    get_gemini_client,
    generate_content,
    upload_file,
    get_file,
)
from utils.prompts import build_youtube_prompt, clean_json_output
from urllib.parse import urlparse, parse_qs

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

router = APIRouter(prefix="/youtube", tags=["YouTube Analysis"])

# --- Helper Functions (YouTube Related) ---

async def _download_video_with_ytdlp(youtube_url: str, download_dir: str) -> str:
    """Download a YouTube video locally so it can be uploaded to Gemini Files API."""
    output_template = os.path.join(download_dir, "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--format",
        "mp4/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "--output",
        output_template,
    ]

    if YTDLP_COOKIES_FILE:
        cmd.extend(["--cookies", YTDLP_COOKIES_FILE])
    elif YTDLP_COOKIES_FROM_BROWSER:
        cmd.extend(["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER])

    cmd.append(youtube_url)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode().strip() or "yt-dlp failed to download the video.")

    for name in os.listdir(download_dir):
        path = os.path.join(download_dir, name)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError("yt-dlp completed but no downloaded video file was found.")


def _prepare_download_dir() -> str:
    download_dir = os.path.join(DOWNLOADED_VIDEO_DIR, uuid4().hex)
    os.makedirs(download_dir, exist_ok=True)
    return download_dir


async def _wait_for_uploaded_video(client, file_name: str):
    waited = 0
    uploaded = await get_file(client, file_name)
    while getattr(uploaded, "state", None) and getattr(uploaded.state, "name", None) == "PROCESSING":
        if waited >= VIDEO_FILE_MAX_WAIT_SECONDS:
            raise TimeoutError("Timed out waiting for Gemini to process the uploaded video.")
        await asyncio.sleep(VIDEO_FILE_POLL_INTERVAL_SECONDS)
        waited += VIDEO_FILE_POLL_INTERVAL_SECONDS
        uploaded = await get_file(client, file_name)

    state_name = getattr(getattr(uploaded, "state", None), "name", None)
    if state_name and state_name != "ACTIVE":
        raise RuntimeError(f"Uploaded video is not usable. Gemini file state: {state_name}")
    return uploaded


async def _run_downloaded_video_analysis(youtube_urls: List[str]) -> Dict[str, Any]:
    """Download YouTube videos, upload them to Gemini Files API, and analyze the resulting media."""
    client = get_gemini_client()
    prompt = build_youtube_prompt()
    uploaded_files = []
    downloaded_paths = []
    download_dir = _prepare_download_dir()

    for url in youtube_urls:
        logging.info("Downloading YouTube video for Gemini upload: %s", url)
        downloaded_path = await _download_video_with_ytdlp(url, download_dir)
        logging.info("Downloaded YouTube video stored at: %s", downloaded_path)
        downloaded_paths.append(downloaded_path)
        uploaded = await upload_file(client, downloaded_path)
        uploaded = await _wait_for_uploaded_video(client, uploaded.name)
        uploaded_files.append(uploaded)

    parts = []
    for uploaded in uploaded_files:
        parts.append(
            types.Part.from_uri(
                file_uri=uploaded.uri,
                mime_type=getattr(uploaded, "mime_type", None) or "video/mp4",
            )
        )
    parts.append(types.Part.from_text(text=prompt))
    contents = [types.Content(role="user", parts=parts)]
    text, req_toks, resp_toks, total_toks = await generate_content(client, MODEL_NAME, contents)

    try:
        cleaned = clean_json_output(text)
        parsed_response = json.loads(cleaned)
    except Exception:
        logging.error("Invalid JSON from model for downloaded video analysis, returning raw text.")
        parsed_response = {"raw_output": text}

    return {
        "youtube_urls": youtube_urls,
        "download_dir": download_dir,
        "downloaded_video_paths": downloaded_paths,
        "model": MODEL_NAME,
        "request_tokens": req_toks,
        "response_tokens": resp_toks,
        "total_tokens": total_toks,
        "analysis": parsed_response,
    }


async def run_youtube_analysis(youtube_urls: List[str]) -> Dict[str, Any]:
    """Analyze YouTube videos by downloading them locally and uploading the media to Gemini."""
    try:
        return await _run_downloaded_video_analysis(youtube_urls)
    except Exception:
        try:
            logging.warning("Downloaded video analysis failed; falling back to transcript-based analysis.\n%s", traceback.format_exc())
            return await run_youtube_transcript_analysis(youtube_urls)
        except Exception:
            raise


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
