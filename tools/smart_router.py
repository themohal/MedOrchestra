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
ACK_TERMS = {
    "ok", "okay", "k", "kk", "great", "thanks", "thank you", "thankyou", "ty",
    "cool", "nice", "got it", "gotcha", "understood", "sure", "fine", "good",
    "great thanks", "okay great", "ok great", "okay thanks", "ok thanks",
    "alright", "perfect", "awesome", "noted", "yes", "no", "yep", "nope", "yeah",
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
    has_prior_report: bool = False,
) -> RouteDecision:
    heuristic = _heuristic_route(
        prompt, transcript, uploaded_files, extracted_documents, age, duration, has_prior_report
    )
    if os.getenv("SMART_ROUTER_MODE", "heuristic").lower() != "llm":
        return heuristic

    llm_decision = _llm_route(prompt, transcript, uploaded_files, extracted_documents, age, duration)
    return llm_decision or heuristic


def _is_acknowledgement(message_lower: str) -> bool:
    """True for short social/acknowledgement replies that shouldn't trigger analysis."""
    cleaned = message_lower.strip().strip(".!?, ")
    if not cleaned:
        return True
    if cleaned in ACK_TERMS:
        return True
    # Short phrases (<= 4 words) with no analysis keyword are treated as chatter.
    words = cleaned.split()
    if len(words) <= 4 and not any(term in cleaned for term in ANALYSIS_TERMS):
        return all(word in ACK_TERMS or len(word) <= 3 for word in words)
    return False


def _heuristic_route(
    prompt: str,
    transcript: str,
    uploaded_files: list[str],
    extracted_documents: str,
    age: int | None,
    duration: str,
    has_prior_report: bool = False,
) -> RouteDecision:
    # Route on the CURRENT turn only. Using the full transcript/extracted_documents
    # here caused every follow-up (even "okay, thanks") to re-trigger the crew,
    # because the accumulated history always looked like medical context.
    message = prompt.strip()
    message_lower = message.lower()
    has_files = bool(uploaded_files)
    has_images = any(Path(path).suffix.lower() in IMAGE_SUFFIXES for path in uploaded_files)

    # Emergency language always takes priority, regardless of conversation state.
    if any(term in message_lower for term in EMERGENCY_TERMS):
        return RouteDecision(
            route=Route.EMERGENCY,
            reason="Potential emergency red-flag language detected in this message.",
            use_crew=True,
            use_vision=has_images,
            use_research=False,
        )

    # Once a case has been analyzed, any follow-up WITHOUT a new file is normal
    # chat. The generated report and uploaded-file text remain in the transcript,
    # so the chat reply can answer questions about the existing report.
    if has_prior_report and not has_files:
        return RouteDecision(
            route=Route.CHAT,
            reason="A report already exists and no new file was attached; answering from the existing report context.",
        )

    # Short acknowledgements / social replies never need analysis.
    if not has_files and _is_acknowledgement(message_lower):
        return RouteDecision(
            route=Route.CHAT,
            reason="Short acknowledgement or follow-up chat; no new analysis needed.",
        )

    asks_for_analysis = any(term in message_lower for term in ANALYSIS_TERMS)
    has_medical_context = len(message) > 40

    # Only run the full crew when this turn actually brings something to analyze:
    # a new file, an explicit analysis request, or a substantive symptom description.
    if has_files or asks_for_analysis or has_medical_context:
        missing_fields = []
        if not age:
            missing_fields.append("age")
        if not duration:
            missing_fields.append("symptom duration")
        return RouteDecision(
            route=Route.ANALYZE_CASE,
            reason="This message includes a new file, an analysis request, or substantive medical context.",
            use_crew=True,
            use_vision=has_images,
            use_research=True,
            missing_fields=missing_fields,
        )

    return RouteDecision(
        route=Route.CHAT,
        reason="General chat or clarification; no new case details to analyze.",
    )


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


def generate_lite_chat_reply(
    prompt: str,
    transcript: str,
    missing_fields: list[str] | None = None,
    report_context: str = "",
) -> str:
    if missing_fields:
        fields = ", ".join(missing_fields)
        return f"Before I analyze this safely, please share: {fields}."

    system_content = (
        "You are a cautious healthcare assistant. Do not diagnose. "
        "Answer briefly and encourage professional care for serious symptoms."
    )
    if report_context:
        system_content += (
            " A specialist analysis report has already been generated for this "
            "conversation (provided below). When the user asks about the report, "
            "their results, the uploaded files, or next steps, answer using that "
            "report context. Do not invent findings that are not in it.\n\n"
            f"=== EXISTING REPORT CONTEXT ===\n{report_context[-6000:]}"
        )

    try:
        from litellm import completion

        response = completion(
            model=os.getenv("LITELLM_CHAT_MODEL", "openai/gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Conversation:\n{transcript[-2500:]}\n\nLatest message:\n{prompt}"},
            ],
            temperature=0.2,
            max_tokens=350,
        )
        return response.choices[0].message.content
    except Exception:
        if report_context:
            return (
                "I have the earlier analysis report in context, but the chat model is "
                "currently unavailable. Please re-check your API key/configuration, or "
                "download the PDF report above."
            )
        return (
            "I can help collect symptoms, review uploaded reports, and prepare a specialist-agent summary. "
            "Describe the concern and attach any reports or images when you are ready."
        )
