from typing import Dict, Any, List, Optional
from pydantic import BaseModel, HttpUrl


class AnalyzeRequest(BaseModel):
    repo_url: HttpUrl


class AnalyzeResponse(BaseModel):
    repo_url: HttpUrl
    request_tokens: int
    response_tokens: int
    total_tokens: int
    model: str
    analysis: Dict[str, Any]


class SemanticAnalyzeResponse(BaseModel):
    repo_url: HttpUrl
    query: str
    response: Dict[str, Any]
    model: str
    request_tokens: int
    response_tokens: int
    total_tokens: int


class YoutubeAnalyzeRequest(BaseModel):
    youtube_urls: List[str]

class YoutubeAnalyzeResponse(BaseModel):
    youtube_urls: List[str]
    model: str
    request_tokens: int
    response_tokens: int
    total_tokens: int
    analysis: Dict[str, Any]
