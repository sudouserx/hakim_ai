"""
Layer 6 — Pathologist UI: HTML Report Renderer.

Generates a self-contained HTML report with:
  - Diagnosis summary card
  - Biomarker dashboard with colour-coded confidence bars
  - Importance map grid
  - Patch evidence gallery
  - NL explanation sections
  - Uncertainty flags panel
  - Actionable recommendations

Design intent: the HTML is standalone (no external JS CDN required),
opens in any browser, and is printable for MDT meetings.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from hakim_ai.types import PathologyReport, PipelineResult
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer6.ui")

# ---------------------------------------------------------------------------
# Colour palette (WHO-inspired clinical palette)
# ---------------------------------------------------------------------------

_COLOURS = {
    "malignant": "#c0392b",
    "suspicious": "#e67e22",
    "benign": "#27ae60",
    "unknown": "#7f8c8d",
    "MSI-H": "#8e44ad",
    "MSS": "#2980b9",
    "positive": "#c0392b",
    "negative": "#27ae60",
    "equivocal": "#e67e22",
    "confidence_high": "#27ae60",
    "confidence_mid": "#f39c12",
    "confidence_low": "#c0392b",
    "background": "#f8f9fa",
    "header": "#2c3e50",
    "text": "#34495e",
}


def _conf_colour(confidence: float) -> str:
    if confidence >= 0.70:
        return _COLOURS["confidence_high"]
    if confidence >= 0.45:
        return _COLOURS["confidence_mid"]
    return _COLOURS["confidence_low"]


def _bar(value: float, colour: str = "#3498db", width: int = 200) -> str:
    """Render a horizontal confidence bar."""
    pct = int(value * 100)
    bar_w = int(value * width)
    return (
        f'<div style="display:inline-flex;align-items:center;gap:6px;">'
        f'<div style="width:{width}px;background:#ddd;border-radius:4px;height:10px;">'
        f'<div style="width:{bar_w}px;background:{colour};height:10px;border-radius:4px;"></div>'
        f'</div><span style="font-size:12px;color:{colour};font-weight:bold;">{pct}%</span></div>'
    )


def _flag_pill(text: str) -> str:
    return (
        f'<span style="background:#fff3cd;border:1px solid #ffc107;color:#856404;'
        f'padding:2px 8px;border-radius:12px;font-size:11px;margin:2px;display:inline-block;">'
        f'⚠ {text}</span>'
    )


class UIRenderer:
    """
    Generates a self-contained HTML pathologist report.

    Inputs:  PipelineResult
    Outputs: HTML string (also saved to output_dir if provided)
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir

    def render(self, result: PipelineResult) -> str:
        """Return the full HTML report as a string."""
        if result.report is None:
            return self._render_error(result)
        return self._render_full(result)

    def save(self, result: PipelineResult) -> str:
        """Render and save the HTML report; return the file path."""
        html = self.render(result)
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(
            self.output_dir, f"report_{result.patient_id}.html"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML report saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def _render_full(self, result: PipelineResult) -> str:
        report = result.report
        diag = report.diagnosis
        mol = report.molecular_predictions
        expl = report.explanation
        conf = diag.overall_confidence
        label_colour = _COLOURS.get(diag.diagnostic_label.value, _COLOURS["unknown"])

        flags_html = (
            "".join(_flag_pill(f) for f in report.uncertainty_flags)
            if report.uncertainty_flags
            else '<span style="color:#27ae60;">✓ No uncertainty flags</span>'
        )
        recs_html = "\n".join(
            f'<li>{rec}</li>' for rec in report.recommendations
        )
        citations_html = "\n".join(
            f'<li style="margin-bottom:6px;">{c}</li>'
            for c in expl.evidence_citations
        )
        differentials_html = "\n".join(
            f'<li>{d}</li>' for d in diag.differential_diagnoses
        )
        features_html = ", ".join(expl.key_morphological_features) or "—"
        biomarker_rows = "".join(
            f'<tr><td style="padding:6px 10px;font-weight:600;">{k}</td>'
            f'<td style="padding:6px 10px;">{v}</td></tr>'
            for k, v in report.biomarker_summary.items()
        )
        concepts_html = "\n".join(
            f'<li>{c}</li>' for c in expl.concept_alignments
        )

        # Importance map grid (16×16 cells → 8px each)
        importance_map_html = self._render_importance_map(result)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Pathology Report — {report.patient_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: {_COLOURS['background']}; color: {_COLOURS['text']}; padding: 24px; }}
  .header {{ background: {_COLOURS['header']}; color: white; padding: 20px 28px;
             border-radius: 8px; margin-bottom: 20px; }}
  .header h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .header p {{ font-size: 13px; opacity: 0.8; }}
  .card {{ background: white; border-radius: 8px; padding: 20px;
           margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .card h2 {{ font-size: 16px; color: {_COLOURS['header']}; margin-bottom: 14px;
              border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  .label-badge {{ display: inline-block; padding: 4px 14px; border-radius: 20px;
                  font-size: 13px; font-weight: 700; color: white;
                  background: {label_colour}; text-transform: uppercase; }}
  table.bio {{ border-collapse: collapse; width: 100%; }}
  table.bio td {{ border-bottom: 1px solid #f0f0f0; font-size: 13px; }}
  table.bio tr:last-child td {{ border-bottom: none; }}
  ul.recs {{ padding-left: 18px; font-size: 13px; line-height: 1.9; }}
  p.narrative {{ font-size: 13px; line-height: 1.7; white-space: pre-wrap; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .uncertainty-box {{ background: #fff9e6; border-left: 4px solid #f39c12;
                      padding: 12px; border-radius: 4px; font-size: 13px; line-height:1.6; }}
  .counterfactual {{ background: #eaf4fb; border-left: 4px solid #3498db;
                     padding: 12px; border-radius: 4px; font-size: 13px; line-height: 1.6; }}
  .footer {{ text-align: center; font-size: 11px; color: #aaa; margin-top: 24px; }}
  @media print {{ body {{ padding: 8px; }} .card {{ box-shadow: none; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>🔬 AI Histopathology Report — Gastric Cancer (STAD)</h1>
  <p>Patient: <strong>{report.patient_id}</strong> &nbsp;|&nbsp;
     Generated: {report.timestamp} &nbsp;|&nbsp;
     Report v{report.report_version} &nbsp;|&nbsp;
     <strong style="color:#e74c3c;">⚠ AI-assisted — requires pathologist sign-off</strong></p>
</div>

<div class="two-col">
  <div class="card">
    <h2>Primary Diagnosis</h2>
    <p style="margin-bottom:10px;">{diag.primary_diagnosis}</p>
    <div style="margin-bottom:10px;"><span class="label-badge">{diag.diagnostic_label.value}</span></div>
    <p style="font-size:12px;">WHO Classification: <strong>{report.who_classification}</strong></p>
    <p style="font-size:12px;">Lauren Type: <strong>{report.lauren_classification.value.capitalize()}</strong></p>
    {f'<p style="font-size:12px;">Grade: <strong>{diag.grade}</strong></p>' if diag.grade else ''}
    {f'<p style="font-size:12px;">TNM (estimated): <strong>{diag.tnm_contribution}</strong></p>' if diag.tnm_contribution else ''}
    <div style="margin-top:10px;">
      Overall Confidence: {_bar(conf, _conf_colour(conf))}
    </div>
  </div>

  <div class="card">
    <h2>Biomarker Predictions (H&amp;E only)</h2>
    <table class="bio">
      {biomarker_rows}
    </table>
  </div>
</div>

<div class="card">
  <h2>Differential Diagnoses</h2>
  <ul class="recs">{differentials_html}</ul>
</div>

<div class="card">
  <h2>AI Explanation</h2>
  <p class="narrative">{expl.narrative}</p>
  {f'<div class="counterfactual" style="margin-top:14px;"><strong>Counterfactual:</strong> {expl.counterfactual_note}</div>' if expl.counterfactual_note else ''}
</div>

<div class="two-col">
  <div class="card">
    <h2>Key Morphological Features</h2>
    <p style="font-size:13px;">{features_html}</p>
    <h2 style="margin-top:14px;">WHO Concept Alignment</h2>
    <ul class="recs">{concepts_html}</ul>
  </div>

  <div class="card">
    <h2>Uncertainty Assessment</h2>
    <div class="uncertainty-box">{expl.uncertainty_statement}</div>
    <div style="margin-top:12px;">{flags_html}</div>
  </div>
</div>

<div class="two-col">
  <div class="card">
    <h2>Evidence Citations</h2>
    <ul class="recs" style="list-style-type:none;padding-left:0;">{citations_html}</ul>
  </div>

  <div class="card">
    <h2>Slide Importance Map</h2>
    {importance_map_html}
    <p style="font-size:11px;color:#888;margin-top:8px;">
      Each cell represents a WSI region; colour intensity = diagnostic importance.
    </p>
  </div>
</div>

<div class="card">
  <h2>Recommendations</h2>
  <ul class="recs">{recs_html}</ul>
</div>

<div class="footer">
  <p>Generated by hakim_ai v{report.report_version} — For research use only.
     Not approved for clinical diagnostic decisions without pathologist oversight.
  </p>
</div>
</body></html>"""

    def _render_importance_map(self, result: PipelineResult) -> str:
        if result.evidence is None or result.evidence.navigation.importance_map is None:
            return "<p style='font-size:12px;color:#aaa;'>No importance map available.</p>"
        grid = result.evidence.navigation.importance_map
        cell_size = 10
        cells: List[str] = []
        for row in grid:
            for val in row:
                intensity = int(val * 255)
                # Red channel for importance (low=blue, high=red)
                r, g, b = intensity, 50, 255 - intensity
                cells.append(
                    f'<div style="width:{cell_size}px;height:{cell_size}px;'
                    f'background:rgb({r},{g},{b});display:inline-block;"></div>'
                )
        cols = len(grid[0]) if grid else 16
        return (
            f'<div style="line-height:0;width:{cols * cell_size}px;">'
            + "".join(cells)
            + "</div>"
        )

    def _render_error(self, result: PipelineResult) -> str:
        return f"""<!DOCTYPE html><html><body>
<h2>Pipeline Error — Patient {result.patient_id}</h2>
<p>Error: {result.error or "Unknown error"}</p>
<p>QC passed: {result.qc_result.passed if result.qc_result else "N/A"}</p>
</body></html>"""