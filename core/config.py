import os
import logging
from dotenv import load_dotenv


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MODEL_NAME = "gemini-3-flash-preview"


def setup_environment() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


# initialize on import
setup_environment()
