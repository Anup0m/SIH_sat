"""
SatQuery AI — Mission Report Generator (Phase 5)
Generates audit-ready, ISRO-standard analytical mission reports from AnalysisResult.
Formats include structured JSON export and human-readable analytical Markdown/HTML.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

from satquery_core.schemas import AnalysisResult


class MissionReportGenerator:
    """Generates structured analytical reports matching SIH Requirement F."""

    @staticmethod
    def generate_markdown_report(result: AnalysisResult, report_id: str = None) -> str:
        rep_id = report_id or f"SATQUERY-RPT-{int(time.time())}"
        date_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        evidence_section = "None"
        if result.evidence:
            ev = result.evidence
            if ev.evidence_type == "bounding_boxes":
                box_lines = "\n".join([f"- **Object {i+1}**: {ev.labels[i] if ev.labels else 'Feature'} at `{box}`" for i, box in enumerate(ev.bounding_boxes or [])])
                evidence_section = f"**Type**: Spatial Bounding Boxes\n{box_lines}"
            elif ev.evidence_type == "change_mask":
                evidence_section = (
                    f"**Type**: Bi-Temporal Change Mask\n"
                    f"- **Surface Area Changed**: {ev.spatial_metadata.get('change_percentage', 'N/A')}%\n"
                    f"- **Generated Mask**: `{ev.mask_path or 'In-memory buffer'}`"
                )

        trace_rows = []
        for step in result.execution_trace:
            tool_info = f" ({step.tool_used})" if step.tool_used else ""
            trace_rows.append(
                f"| {step.step_number} | **{step.step_name}**{tool_info} | {step.action_taken} | `{step.output_summary}` |"
            )
        trace_table = "\n".join(trace_rows)

        report = f"""# SatQuery AI — Remote Sensing Analytical Mission Report
**Report ID**: `{rep_id}`  
**Generated At**: `{date_str}`  
**Engine Version**: `{result.model_version}`  
**Execution Latency**: `{result.processing_time_ms} ms`

---

## 1. Mission Query
> **Query**: *"{result.query}"*

---

## 2. Query Solution (Plain Language / Layman's Summary)
> **Direct Non-Technical Answer**:  
> {result.plain_language_solution or 'Direct ground feature evaluation complete.'}

---

## 3. Technical Intelligence & Detailed Analytics
{result.answer}

* **Confidence Assessment**: **{result.confidence_level}** ({result.confidence_score * 100:.1f}%)

---

## 4. Visual & Spatial Evidence
{evidence_section}

---

## 5. Auditable Execution Trace (SIH Requirement E & F)
| Step | Action | Description | Output Summary |
| :---: | :--- | :--- | :--- |
{trace_table}

---
*Report certified by SatQuery AI Automated Vision-Language Orchestration Engine.*
"""
        return report

    @classmethod
    def generate_geojson(cls, result: AnalysisResult) -> Dict[str, Any]:
        """Generates standard RFC 7946 GeoJSON FeatureCollection for GIS software (QGIS/ArcGIS)."""
        features = []
        if result.evidence and result.evidence.bounding_boxes:
            for i, box in enumerate(result.evidence.bounding_boxes):
                ymin, xmin, ymax, xmax = box
                # Coordinate polygon [longitude, latitude]
                coords = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                    [xmin, ymin]
                ]
                label = result.evidence.labels[i] if (result.evidence.labels and len(result.evidence.labels) > i) else f"Target_{i+1}"
                features.append({
                    "type": "Feature",
                    "properties": {
                        "id": i + 1,
                        "label": label,
                        "confidence": result.confidence_score,
                        "task": "spatial_grounding"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                })

        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features
        }

    @classmethod
    def save_report(cls, result: AnalysisResult, output_dir: Path, report_id: str = None) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rep_id = report_id or f"report_{int(time.time())}"
        
        md_content = cls.generate_markdown_report(result, report_id=rep_id)
        md_file = output_dir / f"{rep_id}.md"
        json_file = output_dir / f"{rep_id}.json"
        geojson_file = output_dir / f"{rep_id}.geojson"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)

        geojson_data = cls.generate_geojson(result)
        with open(geojson_file, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        return {
            "markdown_report": str(md_file),
            "json_report": str(json_file),
            "geojson_report": str(geojson_file)
        }
