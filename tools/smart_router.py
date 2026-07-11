from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Route(str, Enum):
    CHAT = "chat"
    ASK_MISSING_INFO = "ask_missing_info"
    ANALYZE_CASE = "analyze_case"
    EMERGENCY = "emergency"


@dataclass
class RouteDecision:
    route: Route
    reason: str
    use_crew: bool = False
    use_vision: bool = False
    use_research: bool = False
    missing_fields: list[str] = field(default_factory=list)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ANALYSIS_TERMS = {
    "analyze",
    "analysis",
    "diagnose",
    "diagnosis",
    "report",
    "xray",
    "x-ray",
    "scan",
    "lab",
    "blood",
    "prescription",
    "medicine",
    "result",
    "review",
    "check",
}
EMERGENCY_TERMS = {
    "chest pain",
    "shortness of breath",
    "can't breathe",
    "cannot breathe",
    "stroke",
    "face drooping",
    "severe bleeding",
    "unconscious",
    "suicidal",
    "suicide",
    "anaphylaxis",
    "severe allergic",
}


def route_message(
    prompt: str,
    transcript: str,
    uploaded_files: list[str],
    extracted_documents: str,
    age: int | None,
    duration: str,
) -> RouteDecision:
    heuristic = _heuristic_route(prompt, transcript, uploaded_files, extracted_documents, age, duration)
    if os.getenv("SMART_ROUTER_MODE", "heuristic").lower() != "llm":
        return heuristic

    llm_decision = _llm_route(prompt, transcript, uploaded_files, extracted_documents, age, duration)
    return llm_decision or heuristic


def _heuristic_route(
    prompt: str,
    transcript: str,
    uploaded_files: list[str],
    extracted_documents: str,
    age: int | None,
    duration: str,
) -> RouteDecision:
    text = f"{prompt}\n{transcript}\n{extracted_documents}".lower()
    has_files = bool(uploaded_files)
    has_images = any(Path(path).suffix.lower() in IMAGE_SUFFIXES for path in uploaded_files)

    if any(term in text for term in EMERGENCY_TERMS):
        return RouteDecision(
            route=Route.EMERGENCY,
            reason="Potential emergency red-flag language detected.",
            use_crew=True,
            use_vision=has_images,
            use_research=False,
        )

    missing_fields = []
    if not age:
        missing_fields.append("age")
    if not duration:
        missing_fields.append("symptom duration")

    asks_for_analysis = any(term in text for term in ANALYSIS_TERMS)
    has_medical_context = len(text.strip()) > 40 or has_files
    if has_files or asks_for_analysis or has_medical_context:
        return RouteDecision(
            route=Route.ANALYZE_CASE,
            reason="The message includes enough medical context, an analysis request, or uploaded files.",
            use_crew=True,
            use_vision=has_images,
            use_research=True,
            missing_fields=missing_fields,
        )

    if missing_fields:
        return RouteDecision(
            route=Route.ASK_MISSING_INFO,
            reason="The message is short and basic patient context is missing.",
            missing_fields=missing_fields,
        )

    return RouteDecision(route=Route.CHAT, reason="General chat or clarification.")


def _llm_route(
    prompt: str,
    transcript: str,
    uploaded_files: list[str],
    extracted_documents: str,
    age: int | None,
    duration: str,
) -> RouteDecision | None:
    try:
        from litellm import completion

        response = completion(
            model=os.getenv("LITELLM_ROUTER_MODEL", os.getenv("LITELLM_CHAT_MODEL", "openai/gpt-4o-mini")),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify this healthcare chat turn. Return only JSON with keys: "
                        "route, reason, use_crew, use_vision, use_research, missing_fields. "
                        "route must be one of chat, ask_missing_info, analyze_case, emergency."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "prompt": prompt,
                            "transcript": transcript[-2500:],
                            "uploaded_files": uploaded_files,
                            "has_extracted_documents": bool(extracted_documents.strip()),
                            "age": age,
                            "duration": duration,
                        }
                    ),
                },
            ],
            temperature=0,
            max_tokens=250,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return RouteDecision(
            route=Route(data.get("route", "chat")),
            reason=str(data.get("reason", "LiteLLM router decision.")),
            use_crew=bool(data.get("use_crew", False)),
            use_vision=bool(data.get("use_vision", False)),
            use_research=bool(data.get("use_research", False)),
            missing_fields=list(data.get("missing_fields", [])),
        )
    except Exception:
        return None


def generate_lite_chat_reply(prompt: str, transcript: str, missing_fields: list[str] | None = None) -> str:
    if missing_fields:
        fields = ", ".join(missing_fields)
        return f"Before I analyze this safely, please share: {fields}."

    try:
        from litellm import completion

        response = completion(
            model=os.getenv("LITELLM_CHAT_MODEL", "openai/gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious healthcare assistant. Do not diagnose. "
                        "Answer briefly and encourage professional care for serious symptoms."
                    ),
                },
                {"role": "user", "content": f"Conversation:\n{transcript[-2500:]}\n\nLatest message:\n{prompt}"},
            ],
            temperature=0.2,
            max_tokens=350,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "I can help collect symptoms, review uploaded reports, and prepare a specialist-agent summary. "
            "Describe the concern and attach any reports or images when you are ready."
        )
