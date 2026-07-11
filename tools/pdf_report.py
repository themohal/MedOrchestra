from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors

from agents.crew import AnalysisResult, CaseInput


# --- Medical teal/blue theme -------------------------------------------------
PRIMARY = colors.HexColor("#0F766E")      # deep teal (headers)
PRIMARY_DARK = colors.HexColor("#134E4A")  # darker teal
ACCENT = colors.HexColor("#0EA5E9")       # sky blue accent
LIGHT_BG = colors.HexColor("#F0FDFA")     # very light teal card background
CARD_BORDER = colors.HexColor("#99F6E4")  # soft teal border
BODY = colors.HexColor("#1F2937")         # near-black slate for body text
MUTED = colors.HexColor("#6B7280")        # grey for captions/footer
WHITE = colors.white


def build_report_pdf(case_id: int, case: CaseInput, result: AnalysisResult) -> bytes:
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=96,   # leave room for the header band drawn in _decorate
        bottomMargin=54,
        title=f"MedOrchestra Case Report #{case_id}",
        author="MedOrchestra",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MOTitle", parent=styles["Title"], textColor=PRIMARY_DARK, fontSize=20,
        spaceAfter=2, leading=24,
    )
    subtitle_style = ParagraphStyle(
        "MOSubtitle", parent=styles["Normal"], textColor=MUTED, fontSize=9.5,
        spaceAfter=10, leading=13,
    )
    heading_style = ParagraphStyle(
        "MOHeading", parent=styles["Heading2"], textColor=PRIMARY, fontSize=14,
        spaceBefore=14, spaceAfter=4, leading=17,
    )
    subheading_style = ParagraphStyle(
        "MOSubheading", parent=styles["Heading3"], textColor=PRIMARY_DARK, fontSize=11.5,
        spaceBefore=8, spaceAfter=2, leading=15,
    )
    normal_style = ParagraphStyle(
        "MOBody", parent=styles["BodyText"], textColor=BODY, fontSize=10, leading=15,
        alignment=TA_LEFT, spaceAfter=4,
    )
    disclaimer_style = ParagraphStyle(
        "MODisclaimer", parent=normal_style, textColor=PRIMARY_DARK, fontSize=9,
        leading=13, backColor=LIGHT_BG, borderColor=CARD_BORDER, borderWidth=1,
        borderPadding=8, spaceAfter=6,
    )

    def section(title: str):
        return [
            Paragraph(title, heading_style),
            HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=1, spaceAfter=6),
        ]

    story = []

    story.append(Paragraph(f"Case Report&nbsp;#{case_id}", title_style))
    story.append(
        Paragraph(
            f"Generated {case.created_at or 'now'} &bull; Coordinated multi-specialist AI support",
            subtitle_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Educational decision-support report only.</b> This is not a diagnosis and not a "
            "substitute for licensed medical care. In an emergency, seek urgent care immediately.",
            disclaimer_style,
        )
    )
    story.append(Spacer(1, 6))

    # --- Patient intake as a styled two-column card --------------------------
    story.extend(section("Patient Intake"))
    intake_rows = [
        ("Age", str(case.age)),
        ("Sex", case.sex or "Not provided"),
        ("Duration", case.duration or "Not provided"),
        ("Symptoms", case.symptoms or "Not provided"),
        ("Known conditions", case.conditions or "Not provided"),
        ("Current medications", case.medications or "Not provided"),
        ("Allergies", case.allergies or "Not provided"),
    ]
    if case.uploaded_files:
        intake_rows.append(("Attached files", "\n".join(_basename(p) for p in case.uploaded_files)))

    table_data = [
        [
            Paragraph(f"<b>{_escape(label)}</b>", ParagraphStyle("k", parent=normal_style, textColor=PRIMARY_DARK)),
            Paragraph(_escape(value).replace("\n", "<br/>"), normal_style),
        ]
        for label, value in intake_rows
    ]
    intake_table = Table(table_data, colWidths=[1.6 * inch, 4.9 * inch], hAlign="LEFT")
    intake_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 1, CARD_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, CARD_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(intake_table)

    # --- Final report --------------------------------------------------------
    story.extend(section("Final Report"))
    story.append(Paragraph(_markdown_to_reportlab(result.final_report), normal_style))

    if result.research_summary:
        story.extend(section("Research Summary"))
        story.append(Paragraph(_markdown_to_reportlab(result.research_summary), normal_style))

    if result.specialist_opinions:
        story.extend(section("Specialist Opinions"))
        for name, content in result.specialist_opinions.items():
            story.append(Paragraph(_escape(name), subheading_style))
            story.append(Paragraph(_markdown_to_reportlab(content), normal_style))
            story.append(Spacer(1, 6))

    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return buffer.getvalue()


# --- Page decoration: header band, vector logo, footer -----------------------
def _decorate(canvas, doc) -> None:
    _draw_header(canvas, doc)
    _draw_footer(canvas, doc)


def _draw_header(canvas, doc) -> None:
    canvas.saveState()
    width, height = doc.pagesize
    band_h = 64

    # header band
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, height - band_h, width, band_h, fill=1, stroke=0)
    # thin accent stripe under the band
    canvas.setFillColor(ACCENT)
    canvas.rect(0, height - band_h - 3, width, 3, fill=1, stroke=0)

    _draw_logo(canvas, x=54, y=height - band_h + 16, size=32)

    # wordmark
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(98, height - band_h + 30, "MedOrchestra")
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#CCFBF1"))
    canvas.drawString(99, height - band_h + 17, "Multi-specialist AI clinical support")

    canvas.restoreState()


def _draw_logo(canvas, x: float, y: float, size: float) -> None:
    """Draw a rounded white tile with a teal medical cross — a self-contained vector logo."""
    canvas.saveState()
    r = size / 2
    cx, cy = x + r, y + r

    # white rounded tile
    canvas.setFillColor(WHITE)
    canvas.roundRect(x, y, size, size, size * 0.22, fill=1, stroke=0)

    # medical cross
    canvas.setFillColor(PRIMARY)
    arm = size * 0.5     # length of the cross span
    thick = size * 0.18  # thickness of each bar
    # vertical bar
    canvas.rect(cx - thick / 2, cy - arm / 2, thick, arm, fill=1, stroke=0)
    # horizontal bar
    canvas.rect(cx - arm / 2, cy - thick / 2, arm, thick, fill=1, stroke=0)

    canvas.restoreState()


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = doc.pagesize

    canvas.setStrokeColor(CARD_BORDER)
    canvas.setLineWidth(0.75)
    canvas.line(54, 42, width - 54, 42)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(54, 30, "Generated by MedOrchestra — confirm all findings with a licensed clinician.")
    canvas.drawRightString(width - 54, 30, f"Page {canvas.getPageNumber()}")

    canvas.restoreState()


def _markdown_to_reportlab(text: str) -> str:
    escaped = _escape(text)
    lines = []
    for line in escaped.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            lines.append(f"<b>{stripped[4:]}</b>")
        elif stripped.startswith("## "):
            lines.append(f"<b>{stripped[3:]}</b>")
        elif stripped.startswith("# "):
            lines.append(f"<b>{stripped[2:]}</b>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            lines.append(f"&bull;&nbsp;{_inline(stripped[2:])}")
        elif not stripped:
            lines.append("")
        else:
            lines.append(_inline(stripped))
    return "<br/>".join(lines)


def _inline(text: str) -> str:
    """Render simple **bold** markdown inline (text is already HTML-escaped)."""
    parts = text.split("**")
    out = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            out.append(f"<b>{part}</b>")
        else:
            out.append(part)
    return "".join(out)


def _basename(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
