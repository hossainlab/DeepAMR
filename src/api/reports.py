"""PDF report generation for DeepAMR predictions."""

import io
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)


def generate_prediction_report(prediction: Dict) -> bytes:
    """Generate a PDF clinical report for an AMR prediction.

    Args:
        prediction: Prediction dict from the database (frontend format).

    Returns:
        PDF file content as bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=6)
    subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=13, spaceBefore=14)
    body_style = styles["Normal"]
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)

    elements: list = []

    # Title
    elements.append(Paragraph("DeepAMR - Antimicrobial Resistance Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 6 * mm))

    # Sample info
    elements.append(Paragraph("Sample Information", section_style))
    sample_data = [
        ["Sample ID", prediction.get("sampleId", "N/A")],
        ["Organism", prediction.get("organism", "Unknown")],
        ["File", prediction.get("fileName", "N/A")],
        ["Date", prediction.get("createdAt", "N/A")],
        ["Overall Risk", (prediction.get("overallRisk") or "N/A").upper()],
    ]
    if prediction.get("model_version"):
        sample_data.append(["Model Version", prediction["model_version"]])
    t = Table(sample_data, colWidths=[120, 350])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6 * mm))

    # Drug resistance table
    results = prediction.get("results") or []
    if results:
        elements.append(Paragraph("Drug Resistance Profile", section_style))

        header = ["Antibiotic", "Status", "Confidence"]
        rows = [header]
        for r in results:
            status = r.get("status", "?")
            conf = r.get("confidence", 0)
            rows.append([
                r.get("antibiotic", r.get("class", "?")),
                "Resistant" if status == "R" else "Susceptible",
                f"{conf * 100:.1f}%",
            ])

        drug_table = Table(rows, colWidths=[200, 120, 100])

        # Color-code status column
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]
        for i, r in enumerate(results, start=1):
            if r.get("status") == "R":
                style_cmds.append(("BACKGROUND", (1, i), (1, i), colors.HexColor("#fee2e2")))
                style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#dc2626")))
            else:
                style_cmds.append(("BACKGROUND", (1, i), (1, i), colors.HexColor("#dcfce7")))
                style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#16a34a")))

        drug_table.setStyle(TableStyle(style_cmds))
        elements.append(drug_table)
        elements.append(Spacer(1, 6 * mm))

    # Summary
    summary = prediction.get("summary")
    if summary:
        elements.append(Paragraph("Summary", section_style))
        elements.append(Paragraph(
            f"Resistant: {summary.get('resistant', 0)} | "
            f"Intermediate: {summary.get('intermediate', 0)} | "
            f"Susceptible: {summary.get('susceptible', 0)}",
            body_style,
        ))
        elements.append(Spacer(1, 4 * mm))

    # Recommendations (if stored)
    recs = prediction.get("recommendations")
    if recs:
        elements.append(Paragraph("Clinical Recommendations", section_style))
        for rec in recs:
            elements.append(Paragraph(f"• {rec}", body_style))
        elements.append(Spacer(1, 4 * mm))

    # Bangladesh context (if stored)
    bd_recs = prediction.get("bangladesh_recommendations")
    if bd_recs:
        elements.append(Paragraph("Bangladesh Clinical Context", section_style))
        for rec in bd_recs:
            elements.append(Paragraph(f"• {rec}", body_style))
        elements.append(Spacer(1, 4 * mm))

    # Disclaimer
    elements.append(Spacer(1, 10 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI model (DeepAMR Advanced DL, "
        "Micro F1: 84.3%, AUC: 98.6%). Results are intended to assist clinical decision-making "
        "and should NOT replace laboratory-confirmed susceptibility testing. "
        "Always consult qualified healthcare professionals before making treatment decisions.",
        disclaimer_style,
    ))

    doc.build(elements)
    return buf.getvalue()
