from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import CURRENCY, DEFAULT_ACTIONS, PART_COST_MULTIPLIER, REPAIR_ACTIONS, SEVERITIES

DISCLAIMER = "Costs are rough estimates from a rule table, not a quote. A technician should confirm every finding."


def part_multiplier(part: str) -> float:
    for key, mult in PART_COST_MULTIPLIER.items():
        if key in part:
            return mult
    return 1.0


def repair_for(damage: str, severity: str):
    return REPAIR_ACTIONS.get(damage, DEFAULT_ACTIONS)[severity]


def build_report(findings) -> dict:
    items, total_low, total_high = [], 0, 0
    for f in findings:
        action, low, high = repair_for(f.damage, f.severity)
        mult = part_multiplier(f.part)
        low, high = round(low * mult), round(high * mult)
        total_low += low
        total_high += high
        items.append(
            {
                "part": f.part,
                "damage": f.damage,
                "severity": f.severity,
                "confidence": f.confidence,
                "area_pct": f.area_pct,
                "action": action,
                "cost_low": low,
                "cost_high": high,
                "images": list(f.images),
            }
        )
    overall = max((i["severity"] for i in items), key=SEVERITIES.index) if items else "none"
    return {
        "items": items,
        "total_low": total_low,
        "total_high": total_high,
        "currency": CURRENCY,
        "overall_severity": overall,
        "disclaimer": DISCLAIMER,
    }


def export_pdf(car_label: str, report: dict, annotated: dict, created_at: str | None = None) -> bytes:
    """Build a PDF report and return its bytes. `annotated` maps image name to JPG bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36,
        title=f"Inspection report - {car_label}",
    )
    styles = getSampleStyleSheet()
    created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    cur = report["currency"]

    story = [
        Paragraph("Car Damage Inspection Report", styles["Title"]),
        Paragraph(f"Vehicle: {escape(car_label)}", styles["Normal"]),
        Paragraph(f"Date: {escape(created_at)}", styles["Normal"]),
        Paragraph(f"Overall severity: {report['overall_severity']}", styles["Normal"]),
        Spacer(1, 12),
    ]

    if report["items"]:
        rows = [["Part", "Damage", "Severity", "Conf.", "Area %", "Action", f"Est. cost ({cur})"]]
        for it in report["items"]:
            rows.append(
                [
                    it["part"], it["damage"], it["severity"], f"{it['confidence']:.2f}",
                    f"{it['area_pct']}", Paragraph(escape(it["action"]), styles["BodyText"]),
                    f"{it['cost_low']} - {it['cost_high']}",
                ]
            )
        rows.append(["", "", "", "", "", "Total", f"{report['total_low']} - {report['total_high']}"])
        table = Table(rows, colWidths=[90, 65, 55, 40, 45, 130, 98], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ]
            )
        )
        story.append(table)
    else:
        story.append(Paragraph("No damage detected.", styles["Normal"]))

    story += [Spacer(1, 8), Paragraph(escape(report["disclaimer"]), styles["Italic"]), Spacer(1, 14)]

    for name, data in annotated.items():
        iw, ih = ImageReader(BytesIO(data)).getSize()
        scale = min(480 / iw, 300 / ih)
        story.append(Paragraph(escape(name), styles["Normal"]))
        story.append(Image(BytesIO(data), width=iw * scale, height=ih * scale))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buf.getvalue()
