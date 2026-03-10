# GitHub & YouTube Analysis API

FastAPI service that uses Gemini to:
- analyze GitHub repositories,
- run semantic (RAG-based) code analysis,
- extract GitHub URLs from PDF/DOCX files and analyze them,
- assess YouTube presentation quality (native video input or transcript fallback).

## Features

- GitHub repo analysis from URL
- Semantic repo analysis with query + embeddings (FAISS + HuggingFace)
- Document upload (`.pdf` / `.docx`) to extract a GitHub repo URL
- Optional image screenshots (up to 4) for repo context
- YouTube analysis by downloading videos locally, uploading to Gemini, then transcript fallback if needed
- Token usage in API responses (`request_tokens`, `response_tokens`, `total_tokens`)

## Project Structure

```text
IIMBX/
├── app.py
├── core/
│   ├── config.py
│   ├── gemini_client.py
│   └── schemas.py
├── routers/
│   ├── github_router.py
│   └── youtube_router.py
├── utils/
│   ├── doc_extract_utils.py
│   ├── embed_utils.py
│   ├── prompts.py
│   └── repo_utils.py
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- `git` installed (used for cloning repositories during analysis)
- Gemini API key
- Internet access for:
  - Gemini API calls
  - cloning public GitHub repositories
  - first-time HuggingFace embedding model download (`intfloat/multilingual-e5-base`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

## Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

Open:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Endpoints

### Generic

- `GET /` - basic API info
- `GET /health` - health check
- `POST /test-gemini` - test Gemini with a free-form prompt

Example:

```bash
curl -X POST "http://localhost:8001/test-gemini" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Return a JSON object with status ok"}'
```

### GitHub

- `POST /github/analyze` - direct repository analysis
- `POST /github/semantic-analyze` - semantic analysis with query parameter
- `POST /github/analyze-doc` - extract GitHub URL from uploaded PDF/DOCX and analyze

Direct analysis example:

```bash
curl -X POST "http://localhost:8001/github/analyze" \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/psf/requests"}'
```

Semantic analysis example:

```bash
curl -X POST "http://localhost:8001/github/semantic-analyze?query=Explain%20authentication%20flow" \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/psf/requests"}'
```

Document upload example:

```bash
curl -X POST "http://localhost:8001/github/analyze-doc" \
  -F "document=@/path/to/input.pdf" \
  -F "images=@/path/to/screenshot1.png" \
  -F "images=@/path/to/screenshot2.jpg"
```

### YouTube

- `POST /youtube/analyze` - native YouTube analysis
- `POST /youtube/analyze-transcript` - transcript-only analysis

Example:

```bash
curl -X POST "http://localhost:8001/youtube/analyze" \
  -H "Content-Type: application/json" \
  -d '{"youtube_urls":["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]}'
```

## Response Shape (high level)

Most analysis endpoints return:
- source identifiers (`repo_url` or `youtube_urls`)
- `model`
- token usage (`request_tokens`, `response_tokens`, `total_tokens`)
- `analysis` (model-generated JSON object; may include `raw_output` fallback when parsing fails)

## Notes

- `MODEL_NAME` is set in `core/config.py` (currently `gemini-3-flash-preview`).
- `/youtube/analyze` downloads the YouTube video with `yt-dlp`, uploads it to Gemini Files API, and falls back to transcript analysis only if download/upload analysis fails.
- Private YouTube download attempts typically require valid browser cookies. Set `YTDLP_COOKIES_FILE` or `YTDLP_COOKIES_FROM_BROWSER` in `.env` for that use case.
- `/github/analyze-doc` supports `.pdf` and `.docx` only.
