from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agents.crew import CaseInput, run_healthcare_crew
from storage.database import init_db, save_case
from tools.extractors import extract_upload_content
from tools.file_store import save_uploaded_file
from tools.pdf_report import build_report_pdf
from tools.smart_router import Route, generate_lite_chat_reply, route_message
from langsmith.integrations.otel import OtelSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
import dotenv
dotenv.load_dotenv()
# Configure OpenTelemetry
tracer_provider = trace.get_tracer_provider()
if not isinstance(tracer_provider, TracerProvider):
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)

# Add OtelSpanProcessor to the tracer provider
tracer_provider.add_span_processor(OtelSpanProcessor())

# Instrument CrewAI and OpenAI
CrewAIInstrumentor().instrument()
OpenAIInstrumentor().instrument()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DB_PATH = BASE_DIR / "data" / "cases.sqlite3"


def init_session_state() -> None:
    st.session_state.setdefault(
        "messages",
        [
            {
                "role": "assistant",
                "content": (
                    "Tell me what is going on, then attach X-rays, lab reports, prescriptions, "
                    "or PDFs in the upload area below. MedOrchestra will route the case to specialist agents "
                    "and prepare a downloadable PDF report."
                ),
            }
        ],
    )
    st.session_state.setdefault("uploaded_paths", [])
    st.session_state.setdefault("extracted_blocks", [])
    st.session_state.setdefault("processed_upload_names", set())
    st.session_state.setdefault("last_pdf", None)
    st.session_state.setdefault("last_case_id", None)
    st.session_state.setdefault("last_report_text", "")


def render_safety_notice() -> None:
    st.warning(
        "This app provides educational decision support only. It is not a diagnosis, "
        "not a replacement for a licensed doctor, and not for emergencies. If symptoms "
        "are severe or rapidly worsening, seek urgent medical care now."
    )


def main() -> None:
    load_dotenv()
    init_db(DB_PATH)
    init_session_state()

    st.set_page_config(page_title="MedOrchestra", page_icon="+", layout="wide")
    st.title("MedOrchestra")
    st.caption("Coordinated multi-specialist AI support for medical reports, images, and patient questions.")
    render_safety_notice()

    with st.sidebar:
        st.header("Configuration")
        openai_ready = bool(os.getenv("OPENAI_API_KEY"))
        tavily_ready = bool(os.getenv("TAVILY_API_KEY"))
        st.write(f"OpenAI API: {'configured' if openai_ready else 'missing'}")
        st.write(f"Tavily API: {'configured' if tavily_ready else 'missing'}")
        st.caption("Add keys to `.env`, then restart Streamlit.")

        st.header("Patient Context")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Sex", ["Prefer not to say", "Female", "Male", "Intersex"])
        duration = st.text_input("Duration", placeholder="Example: 3 days, 2 weeks")
        conditions = st.text_area("Known conditions", placeholder="Diabetes, asthma, hypertension...")
        medications = st.text_area("Current medications", placeholder="Name, dose, frequency if known")
        allergies = st.text_area("Allergies", placeholder="Medicine or food allergies")

        if st.session_state.last_pdf and st.session_state.last_case_id:
            st.header("Latest Report")
            st.download_button(
                "Download PDF report",
                data=st.session_state.last_pdf,
                file_name=f"medorchestra_case_{st.session_state.last_case_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    st.subheader("Medical Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    submission = st.chat_input(
        "Describe symptoms, ask a follow-up question, or attach reports/images...",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "pdf", "txt"],
        key="medorchestra_chat_input",
    )
    if submission:
        prompt, submitted_files = normalize_chat_submission(submission)
        attached_names = [upload.name for upload in submitted_files]
        user_content = build_user_message_content(prompt, attached_names)
        st.session_state.messages.append({"role": "user", "content": user_content})
        with st.chat_message("user"):
            st.markdown(user_content)

        processed_names = []
        current_paths = []
        if submitted_files:
            with st.chat_message("assistant"):
                processed_names, current_paths = process_chat_uploads(submitted_files)

        transcript = build_user_transcript()
        if not transcript.strip() and not st.session_state.uploaded_paths:
            response = "Please describe the concern or attach a report/image before analysis."
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            return

        decision = route_message(
            prompt=prompt,
            transcript=transcript,
            uploaded_files=[str(path) for path in current_paths],
            extracted_documents="\n\n".join(st.session_state.extracted_blocks),
            age=int(age),
            duration=duration,
            has_prior_report=st.session_state.last_case_id is not None,
        )

        if decision.route == Route.CHAT or decision.route == Route.ASK_MISSING_INFO:
            report_context = build_report_context()
            response = generate_lite_chat_reply(
                prompt, transcript, decision.missing_fields, report_context
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                if processed_names:
                    st.caption("Attached this turn: " + ", ".join(processed_names))
                st.caption(f"Smart route: {decision.route.value} - {decision.reason}")
                st.markdown(response)
            return

        with st.chat_message("assistant"):
            with st.status("Running specialist agents...", expanded=True) as status:
                status.write(f"Smart route: {decision.route.value} - {decision.reason}")
                if decision.use_vision:
                    status.write("Image-aware analysis requested for uploaded medical images.")
                if decision.use_research:
                    status.write("Medical research search enabled for this case.")
                case = CaseInput(
                    age=int(age),
                    sex=sex,
                    symptoms=transcript,
                    duration=duration,
                    conditions=conditions,
                    medications=medications,
                    allergies=allergies,
                    extracted_documents="\n\n".join(st.session_state.extracted_blocks),
                    uploaded_files=[str(path) for path in current_paths],
                    created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                )
                status.write("Checking uploaded reports and images.")
                result = run_healthcare_crew(case)
                status.write("Saving case and building PDF.")
                case_id = save_case(DB_PATH, case, result)
                pdf_bytes = build_report_pdf(case_id, case, result)
                status.update(label="Analysis complete.", state="complete")

            st.markdown(result.final_report)
            with st.expander("Specialist details"):
                for name, content in result.specialist_opinions.items():
                    st.markdown(f"### {name}")
                    st.markdown(content)
            if result.research_summary:
                with st.expander("Research summary"):
                    st.markdown(result.research_summary)
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"medorchestra_case_{case_id}.pdf",
                mime="application/pdf",
            )

        st.session_state.last_pdf = pdf_bytes
        st.session_state.last_case_id = case_id
        st.session_state.last_report_text = _compose_report_context(result)
        st.session_state.messages.append({"role": "assistant", "content": result.final_report})


def process_chat_uploads(uploads) -> tuple[list[str], list]:
    """Process files attached this turn.

    Returns (attached_names, current_paths) for the CURRENT turn only. Extracted
    text is also appended to st.session_state.extracted_blocks so earlier file
    content stays available as conversation context for follow-up questions.
    """
    if not uploads:
        return [], []

    new_uploads = [upload for upload in uploads if upload_key(upload) not in st.session_state.processed_upload_names]
    if not new_uploads:
        return [], []

    with st.status("Attaching files...", expanded=True) as status:
        attached = []
        current_paths = []
        for upload in new_uploads:
            stored_path = save_uploaded_file(upload, UPLOAD_DIR)
            extracted = extract_upload_content(stored_path)
            st.session_state.uploaded_paths.append(stored_path)
            st.session_state.extracted_blocks.append(f"File: {upload.name}\n{extracted}")
            st.session_state.processed_upload_names.add(upload_key(upload))
            attached.append(upload.name)
            current_paths.append(stored_path)
            status.write(f"Attached `{upload.name}`")
        status.update(label="Files attached.", state="complete")

    return attached, current_paths


def build_user_transcript() -> str:
    user_messages = [message["content"] for message in st.session_state.messages if message["role"] == "user"]
    return "\n\n".join(user_messages)


def _compose_report_context(result) -> str:
    """Flatten an analysis result into text the chat model can answer from later."""
    parts = [f"FINAL REPORT:\n{result.final_report}"]
    for name, content in result.specialist_opinions.items():
        parts.append(f"{name.upper()} OPINION:\n{content}")
    if result.research_summary:
        parts.append(f"RESEARCH SUMMARY:\n{result.research_summary}")
    return "\n\n".join(parts)


def build_report_context() -> str:
    """Combine the last generated report with extracted file text for chat grounding."""
    parts = []
    if st.session_state.get("last_report_text"):
        parts.append(st.session_state.last_report_text)
    if st.session_state.get("extracted_blocks"):
        parts.append("UPLOADED FILE CONTENT:\n" + "\n\n".join(st.session_state.extracted_blocks))
    return "\n\n".join(parts)


def upload_key(upload) -> str:
    return f"{upload.name}:{getattr(upload, 'size', 0)}"


def normalize_chat_submission(submission) -> tuple[str, list]:
    if isinstance(submission, str):
        return submission.strip(), []
    text = getattr(submission, "text", "") or submission.get("text", "")
    files = getattr(submission, "files", None)
    if files is None:
        files = submission.get("files", [])
    return text.strip(), list(files or [])


def build_user_message_content(prompt: str, attached_names: list[str]) -> str:
    content = prompt or "Please analyze the attached file(s)."
    if attached_names:
        content += "\n\nAttached: " + ", ".join(f"`{name}`" for name in attached_names)
    return content


if __name__ == "__main__":
    main()
