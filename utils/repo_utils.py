import os
import asyncio
import logging
from typing import List, Dict
import aiofiles
from git import Repo


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
                    "content": content[:max_chars],
                })
            except Exception as e:
                logging.warning(f"Could not read {file_path}: {e}")
    return code_files


def extract_github_links(text: str) -> List[str]:
    import re
    return re.findall(r"https?://github.com/[\w\-]+/[\w\-]+", text)
