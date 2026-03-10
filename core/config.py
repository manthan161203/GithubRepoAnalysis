import os
import logging
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MODEL_NAME = "gemini-3-flash-preview"
VIDEO_FILE_POLL_INTERVAL_SECONDS = int(os.getenv("VIDEO_FILE_POLL_INTERVAL_SECONDS", "5"))
VIDEO_FILE_MAX_WAIT_SECONDS = int(os.getenv("VIDEO_FILE_MAX_WAIT_SECONDS", "300"))
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE")
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
DOWNLOADED_VIDEO_DIR = os.getenv("DOWNLOADED_VIDEO_DIR", "/tmp/iimbx_youtube_downloads")


def setup_environment() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


# initialize on import
setup_environment()
