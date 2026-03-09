# GitHub & YouTube Analysis AI

An advanced analysis tool powered by Gemini AI that evaluates GitHub repositories and YouTube presentation skills.

## 🚀 Features

- **GitHub Repository Analysis**: Direct expert review of repositories using source code.
- **RAG-based Semantic Search**: Analyze specific parts of a repository using natural language queries.
- **Document-to-Repo Analysis**: Extract GitHub URLs from PDF/DOCX files and analyze them instantly.
- **Visual Feedback**: Supports up to 4 screenshots for enhanced repository review.
- **YouTube Presentation Audit**: Native AI-powered evaluation of body language, tonality, and structure.
- **Transcript-based Fallback**: Reliable analysis even when native video support is unavailable.

---

## 📂 Project Structure

The project follows a clean, modular architecture:

- **`core/`**: Central configuration, models, and Gemini AI client.
- **`utils/`**: Helper utilities for git operations, embeddings, document parsing, and prompts.
- **`routers/`**: Categorized API endpoints for maintainability.
- **`app.py`**: Main entry point for the FastAPI application.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd IIMBX
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```

---

## 📡 API Endpoints

### GitHub Analysis (`/github`)
- **`POST /github/analyze`**: Standard repo analysis.
- **`POST /github/semantic-analyze`**: Query-based code review.
- **`POST /github/analyze-doc`**: Upload PDF/DOCX + optional images.

### YouTube Analysis (`/youtube`)
- **`POST /youtube/analyze`**: Native video analysis.
- **`POST /youtube/analyze-transcript`**: Fallback transcript analysis.

### Generic
- **`GET /health`**: System status.
- **`POST /test-gemini`**: Direct prompt testing with token usage.
- **`GET /`**: API overview.

---

## 🏃 How to Run

Start the FastAPI development server:

```bash
uvicorn app:app --reload --port 8001 --host 0.0.0.0
```

Access the interactive API documentation at `http://localhost:8001/docs`.
