import json
from typing import List, Optional
from core.config import CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME


def build_prompt(repo_url: str, context: str, mode: str = "direct") -> str:
    return f"""
You are an expert software reviewer and UX auditor.

Analyze the GitHub repository: {repo_url}.
Use the provided code context to evaluate.

Return the analysis STRICTLY as a well-formatted JSON object
with the following schema:

{{
  "total_score": <integer 0–100>,
  "section_scores": {{
    "Business Understanding": <integer 0–10>,
    "Objectives Clarity": <integer 0–10>,
    "Design Rationale": <integer 0–10>,
    "Responsiveness": <integer 0–10>,
    "Feature Completeness": <integer 0–20>,
    "Brand Alignment": <integer 0–10>,
    "UX Structure & CTAs": <integer 0–20>,
    "Personal Impression": <integer 0–10>
  }},
  "section_reasoning": {{
    "Business Understanding": <string>,
    "Objectives Clarity": <string>,
    "Design Rationale": <string>,
    "Responsiveness": <string>,
    "Feature Completeness": <string>,
    "Brand Alignment": <string>,
    "UX Structure & CTAs": <string>,
    "Personal Impression": <string>
  }},
  "overall_feedback": <string>
}}

Rules:
- Do not add comments or explanation outside the JSON.
- Output must be valid JSON.
- Base scoring on repository architecture, clarity, design, code quality, and usability.

Repository context (truncated for length):
{context}
"""


def build_youtube_prompt() -> str:
    return """
You are an expert communication coach and presentation evaluator.

Watch the video carefully and evaluate the speaker's presentation using the following rubric.
Each criterion is scored on a scale of 1 to 5:
  5 = Excellent
  4 = Good
  3 = Average
  2 = Satisfactory
  1 = Poor

Criteria to evaluate:
1. Body Language  — posture, gestures, movement, use of space
2. Facial Expression — eye contact, expressiveness, engagement, smile
3. Tonality — voice modulation, pace, pitch, clarity, energy
4. Structure — logical flow, introduction, body, conclusion, transitions
5. Impact — persuasiveness, audience engagement, memorability, overall impression

Return ONLY a valid JSON object in this exact schema (no explanation, no markdown):

{
  "total_score": <integer, sum of all 5 criteria, max 25>,
  "scores": {
    "Body Language": <integer 1-5>,
    "Facial Expression": <integer 1-5>,
    "Tonality": <integer 1-5>,
    "Structure": <integer 1-5>,
    "Impact": <integer 1-5>
  },
  "reasoning": {
    "Body Language": <string, 1-2 sentence justification>,
    "Facial Expression": <string, 1-2 sentence justification>,
    "Tonality": <string, 1-2 sentence justification>,
    "Structure": <string, 1-2 sentence justification>,
    "Impact": <string, 1-2 sentence justification>
  },
  "overall_feedback": <string, 3-4 sentence holistic summary of the speaker's performance>,
  "strengths": [<string>, ...],
  "areas_for_improvement": [<string>, ...]
}

Rules:
- Output must be valid JSON only.
- Do not include any text before or after the JSON object.
- All scores must be integers from 1 to 5.
- total_score must equal the sum of all 5 individual scores.
"""


def clean_json_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 1)[-1]
        if text.lower().startswith("json"):
            text = text[4:].strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text
