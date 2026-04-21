import io
from pathlib import Path
from string import Template

try:
    from xhtml2pdf import pisa
    PDF_GENERATOR_AVAILABLE = True
except Exception as e:
    print(f"Warning: xhtml2pdf unavailable, falling back to HTML. ({e})")
    PDF_GENERATOR_AVAILABLE = False

TEMPLATE_PATH = Path(__file__).parent / "template.html"

def _render_html(report: dict) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    defect = report.get("defect", {})
    analysis = report.get("analysis", {})
    decision = report.get("decision", {})

    replacements = {
        "{{ report_id }}": report.get("report_id", "N/A"),
        "{{ generated_at }}": report.get("generated_at", "")[:19].replace("T", " "),
        "{{ camera_id }}": report.get("camera_id", "cam0"),
        "{{ line_id }}": report.get("line_id", "line0"),
        "{{ defect_type }}": defect.get("type", "UNKNOWN"),
        "{{ severity }}": defect.get("severity", "UNKNOWN"),
        "{{ severity_lower }}": defect.get("severity", "low").lower(),
        "{{ zone }}": defect.get("zone", "SURFACE"),
        "{{ confidence_pct }}": str(round(defect.get("confidence", 0) * 100, 1)),
        "{{ action }}": decision.get("action", "LOG_ONLY"),
        "{{ timestamp }}": report.get("generated_at", "")[:19].replace("T", " "),
        "{{ cause_hypothesis }}": analysis.get("cause_hypothesis", ""),
        "{{ action_rationale }}": decision.get("rationale", ""),
    }

    # Handle optional image
    frame_path = report.get("frame_path", "")
    if frame_path and Path(frame_path).exists():
        replacements["{% if image_path %}"] = ""
        replacements["{{ image_path }}"] = frame_path
        replacements["{% endif %}"] = ""
    else:
        # Strip the image block
        start = template.find("{% if image_path %}")
        end = template.find("{% endif %}") + len("{% endif %}")
        if start != -1 and end != -1:
            template = template[:start] + template[end:]
        replacements.pop("{% if image_path %}", None)
        replacements.pop("{{ image_path }}", None)
        replacements.pop("{% endif %}", None)

    for k, v in replacements.items():
        template = template.replace(k, str(v))
    return template

def generate_pdf(report: dict) -> bytes:
    """
    Render the defect report as PDF bytes.
    Uses xhtml2pdf if available, else returns rendered HTML bytes as fallback.
    """
    html_content = _render_html(report)

    if PDF_GENERATOR_AVAILABLE:
        result = io.BytesIO()
        pisa_status = pisa.CreatePDF(
            src=html_content, 
            dest=result
        )
        if not pisa_status.err:
            return result.getvalue()
        else:
            print("PDF generation error, falling back to HTML.")

    # Fallback: return HTML bytes (still downloadable, renders in browser)
    return html_content.encode("utf-8")
