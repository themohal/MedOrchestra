from __future__ import annotations

import os
from dataclasses import dataclass, field

from tools.medical_research import trusted_medical_search


SAFETY_INSTRUCTIONS = """
Safety requirements:
- Do not claim certainty or provide a definitive diagnosis.
- Clearly separate likely possibilities, urgent red flags, and suggested next steps.
- Recommend emergency care for chest pain, severe breathing trouble, stroke symptoms,
  severe allergic reaction, severe bleeding, suicidal intent, or altered consciousness.
- For imaging, state that this is not a formal radiology interpretation.
- For medications, tell the user to confirm changes with a licensed clinician.
- Cite research snippets when they are used.
"""


@dataclass
class CaseInput:
    age: int
    sex: str
    symptoms: str
    duration: str
    conditions: str
    medications: str
    allergies: str
    extracted_documents: str
    uploaded_files: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_prompt(self) -> str:
        files = "\n".join(f"- {path}" for path in self.uploaded_files) or "No uploaded files."
        return f"""
Patient:
- Age: {self.age}
- Sex: {self.sex}
- Duration: {self.duration or "Not provided"}
- Symptoms: {self.symptoms or "Not provided"}
- Known conditions: {self.conditions or "Not provided"}
- Current medications: {self.medications or "Not provided"}
- Allergies: {self.allergies or "Not provided"}

Uploaded files:
{files}

Extracted document text:
{self.extracted_documents or "No extracted document text."}
"""


@dataclass
class AnalysisResult:
    final_report: str
    specialist_opinions: dict[str, str]
    research_summary: str = ""


def run_healthcare_crew(case: CaseInput) -> AnalysisResult:
    research = trusted_medical_search(case.symptoms, case.extracted_documents)

    try:
        return _run_crewai(case, research)
    except Exception as exc:
        return _fallback_result(case, research, exc)


def _run_crewai(case: CaseInput, research: str) -> AnalysisResult:
    from crewai import Agent, Crew, Process, Task

    try:
        from crewai import LLM

        llm = LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    except Exception:
        llm = None

    common = f"{SAFETY_INSTRUCTIONS}\n\nCase context:\n{case.to_prompt()}\n\nResearch snippets:\n{research}"

    triage_agent = Agent(
        role="Emergency triage physician",
        goal="Identify urgent red flags and immediate safety guidance.",
        backstory="A cautious clinician who prioritizes emergency escalation and patient safety.",
        llm=llm,
        verbose=False,
    )
    lab_agent = Agent(
        role="Pathology and lab medicine specialist",
        goal="Interpret uploaded report text and flag abnormal or missing lab context.",
        backstory="A lab-focused physician who explains values carefully and avoids inventing reference ranges.",
        llm=llm,
        verbose=False,
    )
    radiology_agent = Agent(
        role="Radiology assistant",
        goal="Review image-related context and describe limitations of AI image review.",
        backstory="A radiology support agent that never replaces a board-certified radiologist.",
        llm=llm,
        verbose=False,
    )
    pharmacology_agent = Agent(
        role="Clinical pharmacology specialist",
        goal="Review medications, allergies, and possible interaction concerns.",
        backstory="A medication safety specialist focused on practical questions for clinicians.",
        llm=llm,
        verbose=False,
    )
    research_agent = Agent(
        role="Medical research specialist",
        goal="Summarize relevant guideline and evidence snippets from trusted sources.",
        backstory="A medical librarian who sticks to reliable sources and explicit citations.",
        llm=llm,
        verbose=False,
    )
    final_agent = Agent(
        role="Primary care coordinator",
        goal="Create a clear patient-friendly final report from all specialist opinions.",
        backstory="A general physician who integrates specialist perspectives into safe next steps.",
        llm=llm,
        verbose=False,
    )

    triage_task = Task(
        description=f"{common}\n\nProvide urgent red flags and triage guidance.",
        expected_output="Triage risk level, red flags, and immediate care advice.",
        agent=triage_agent,
    )
    lab_task = Task(
        description=f"{common}\n\nAnalyze report and lab text if present.",
        expected_output="Lab/report interpretation with missing data and clinician questions.",
        agent=lab_agent,
    )
    radiology_task = Task(
        description=f"{common}\n\nAnalyze uploaded image context. If direct image interpretation is unavailable, explain limitations and next steps.",
        expected_output="Radiology-style observations, limitations, and recommended formal review.",
        agent=radiology_agent,
    )
    pharmacology_task = Task(
        description=f"{common}\n\nAssess medication and allergy concerns.",
        expected_output="Medication safety notes and questions to ask a clinician or pharmacist.",
        agent=pharmacology_agent,
    )
    research_task = Task(
        description=f"{common}\n\nSummarize the trusted research snippets without adding unsupported claims.",
        expected_output="Research summary with source names and URLs where available.",
        agent=research_agent,
    )
    final_task = Task(
        description=f"{common}\n\nCombine all prior work into a patient-friendly report with sections: Safety, Possible Considerations, Uploaded Report Notes, Medication Notes, Recommended Next Steps, Questions for Doctor, Limitations.",
        expected_output="A final patient-friendly report with explicit uncertainty and safety disclaimer.",
        agent=final_agent,
        context=[triage_task, lab_task, radiology_task, pharmacology_task, research_task],
    )

    crew = Crew(
        agents=[triage_agent, lab_agent, radiology_agent, pharmacology_agent, research_agent, final_agent],
        tasks=[triage_task, lab_task, radiology_task, pharmacology_task, research_task, final_task],
        process=Process.sequential,
        verbose=False,
    )
    output = crew.kickoff()

    task_outputs = getattr(output, "tasks_output", []) or []
    names = ["Triage", "Lab Report", "Radiology", "Pharmacology", "Research", "Final"]
    specialist_opinions = {}
    for name, task_output in zip(names, task_outputs):
        specialist_opinions[name] = str(task_output)

    final_report = str(output)
    if specialist_opinions.get("Final"):
        final_report = specialist_opinions["Final"]

    return AnalysisResult(
        final_report=final_report,
        specialist_opinions={k: v for k, v in specialist_opinions.items() if k != "Final"},
        research_summary=specialist_opinions.get("Research", research),
    )


def _fallback_result(case: CaseInput, research: str, exc: Exception) -> AnalysisResult:
    final = f"""
## Analysis could not run

The CrewAI analysis failed locally: `{type(exc).__name__}: {exc}`.

## Safety

This app cannot diagnose or rule out disease. If there is chest pain, severe shortness of breath, stroke-like symptoms, severe bleeding, fainting, suicidal intent, or rapidly worsening symptoms, seek emergency care immediately.

## Case Snapshot

{case.to_prompt()}

## Research Snippets

{research or "No Tavily research snippets were available."}
"""
    return AnalysisResult(
        final_report=final,
        specialist_opinions={"System": "CrewAI execution failed; check API keys, dependencies, and model configuration."},
        research_summary=research,
    )
