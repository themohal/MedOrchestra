from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agents.crew import AnalysisResult, CaseInput


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                case_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )


def save_case(db_path: Path, case: CaseInput, result: AnalysisResult) -> int:
    payload = {
        "age": case.age,
        "sex": case.sex,
        "symptoms": case.symptoms,
        "duration": case.duration,
        "conditions": case.conditions,
        "medications": case.medications,
        "allergies": case.allergies,
        "extracted_documents": case.extracted_documents,
        "uploaded_files": case.uploaded_files,
        "created_at": case.created_at,
    }
    result_payload = {
        "final_report": result.final_report,
        "specialist_opinions": result.specialist_opinions,
        "research_summary": result.research_summary,
    }
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO cases (created_at, case_json, result_json) VALUES (?, ?, ?)",
            (case.created_at, json.dumps(payload), json.dumps(result_payload)),
        )
        return int(cursor.lastrowid)
