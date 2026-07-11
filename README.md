# MedOrchestra

Streamlit MVP for a coordinated multi-specialist healthcare decision-support assistant powered by CrewAI, OpenAI, Tavily search, LiteLLM, SQLite, and local file storage.

This app is for educational and clinical decision-support workflows only. It must not be used as a substitute for a licensed clinician, emergency care, or formal radiology/pathology interpretation.

## Features

- Chat-first medical workspace
- Upload medical images, X-rays, PDFs, text reports, and prescriptions from the chat input attachment button
- Extract text from PDFs and plain text reports
- Optional OCR for images when Tesseract is installed
- Multi-agent analysis with CrewAI
- Tavily-powered medical research snippets from trusted sources
- SQLite case history
- Local file storage under `data/uploads`
- Downloadable PDF report
- Smart routing to avoid running the full specialist crew for every chat message

## Chat Workflow

1. Add optional patient context in the sidebar.
2. Type the patient's concern or follow-up question in the chat input.
3. Optionally attach X-rays, reports, prescriptions, or PDFs from the chat input attachment button.
4. The CrewAI specialist team analyzes the full chat plus uploaded files.
5. Download the generated PDF report from the chat response.

## Smart Routing

The app uses a deterministic router by default:

- `chat`: lightweight response only
- `ask_missing_info`: ask for age, duration, or other needed context
- `analyze_case`: run Tavily research, CrewAI specialists, SQLite save, and PDF export
- `emergency`: immediately flags red-flag language and still routes into the safety-focused crew

To let a LiteLLM-backed model classify chat turns, set:

```text
SMART_ROUTER_MODE=llm
```

Keep the default heuristic mode for local development or when API keys are unavailable.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
OPENAI_MODEL=gpt-4o-mini
```

Run:

```powershell
streamlit run app.py
```

## Notes

- For image OCR, install Tesseract separately and make sure it is available on PATH.
- CrewAI uses the configured OpenAI model through `OPENAI_API_KEY`.
- Tavily is used only for research snippets; the medical agents should cite when they use those snippets.
