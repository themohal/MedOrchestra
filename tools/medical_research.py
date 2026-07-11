from __future__ import annotations

import os


TRUSTED_DOMAINS = [
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "medlineplus.gov",
    "cdc.gov",
    "who.int",
    "nhs.uk",
    "nice.org.uk",
    "fda.gov",
    "radiopaedia.org",
    "mayoclinic.org",
]


def trusted_medical_search(symptoms: str, extracted_documents: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key is not configured; no live medical research was performed."

    query = _build_query(symptoms, extracted_documents)
    if not query:
        return "No searchable symptoms or report text were provided."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=6,
            include_answer=True,
            include_domains=TRUSTED_DOMAINS,
        )
    except Exception as exc:
        return f"Tavily research failed: {exc}"

    lines = []
    answer = response.get("answer")
    if answer:
        lines.append(f"Search answer: {answer}")

    for result in response.get("results", []):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        lines.append(f"- {title}\n  URL: {url}\n  Summary: {content}")

    return "\n\n".join(lines) or "No Tavily results returned from trusted domains."


def _build_query(symptoms: str, extracted_documents: str) -> str:
    source = f"{symptoms}\n{extracted_documents}".strip()
    if not source:
        return ""
    compact = " ".join(source.split())
    return f"medical guideline differential diagnosis patient education {compact[:500]}"
