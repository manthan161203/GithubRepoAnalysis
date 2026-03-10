import asyncio
import os
import logging
import traceback
from typing import Optional
from google import genai
from google.genai import types


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY in environment.")
    return genai.Client(api_key=api_key)


async def count_tokens(client: genai.Client, contents: list, model: str) -> int:
    result = await client.aio.models.count_tokens(model=model, contents=contents)
    return getattr(result, "total_tokens", 0)


async def generate_content(client: genai.Client, model: str, contents: list, config: Optional[types.GenerateContentConfig] = None):
    request_tokens = await count_tokens(client, contents, model)
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config or types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_modalities=["TEXT"],
            ),
        )
    except Exception as e:
        logging.error("Gemini generate_content failed: %s", traceback.format_exc())
        try:
            logging.error("Exception repr: %s", repr(e))
            logging.error("Exception dict: %s", getattr(e, '__dict__', {}))
            logging.error("Exception args: %s", getattr(e, 'args', None))
            logging.error("Exception response attribute: %s", getattr(e, 'response', None))
        except Exception:
            logging.error("Failed to log exception details")
        raise

    response_text = response.text or ""
    usage = getattr(response, "usage_metadata", None)
    response_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    total_tokens = getattr(usage, "total_token_count", request_tokens)
    return response_text, request_tokens, response_tokens, total_tokens


async def upload_file(client: genai.Client, file_path: str):
    return await asyncio.to_thread(client.files.upload, file=file_path)


async def get_file(client: genai.Client, file_name: str):
    return await asyncio.to_thread(client.files.get, name=file_name)
