"""
protocol_automation/parser.py
─────────────────────────────
LLM-powered clinical protocol parser.

Extracts structured study metadata from protocol PDFs including:
- Study title, phase, indication
- Primary and secondary endpoints
- Inclusion / exclusion criteria
- Treatment arms and dose levels
- Visit schedule

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import openai

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ProtocolMetadata:
    """Structured output produced by ProtocolParser."""

    study_title: str = ""
    protocol_number: str = ""
    phase: str = ""
    indication: str = ""
    sponsor: str = ""
    primary_endpoints: list[str] = field(default_factory=list)
    secondary_endpoints: list[str] = field(default_factory=list)
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    treatment_arms: list[dict] = field(default_factory=list)
    visit_schedule: list[dict] = field(default_factory=list)
    raw_text_length: int = 0

    def to_dict(self) -> dict:
        """Serialize to plain dictionary."""
        return {
            "study_title": self.study_title,
            "protocol_number": self.protocol_number,
            "phase": self.phase,
            "indication": self.indication,
            "sponsor": self.sponsor,
            "primary_endpoints": self.primary_endpoints,
            "secondary_endpoints": self.secondary_endpoints,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "treatment_arms": self.treatment_arms,
            "visit_schedule": self.visit_schedule,
            "raw_text_length": self.raw_text_length,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ── Parser ────────────────────────────────────────────────────────────────────

class ProtocolParser:
    """
    Extracts structured metadata from a clinical protocol PDF
    using an LLM (OpenAI GPT-4 by default).

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY environment variable.
    model : str
        LLM model to use. Default is gpt-4o.
    max_pages : int
        Maximum number of PDF pages to send to the LLM.
        Large protocols are truncated to keep within token limits.

    Examples
    --------
    >>> parser = ProtocolParser()
    >>> metadata = parser.parse("path/to/protocol.pdf")
    >>> print(metadata.to_json())
    """

    _SYSTEM_PROMPT = """
You are an expert clinical data scientist with deep knowledge of ICH E6 GCP,
FDA, and EMA regulatory standards. You will be given raw text extracted from
a clinical trial protocol PDF.

Your task is to extract the following information and return it as valid JSON
with exactly these keys:

{
  "study_title": "string",
  "protocol_number": "string",
  "phase": "string  (e.g. Phase 1, Phase 2, Phase 3)",
  "indication": "string",
  "sponsor": "string",
  "primary_endpoints": ["string", ...],
  "secondary_endpoints": ["string", ...],
  "inclusion_criteria": ["string", ...],
  "exclusion_criteria": ["string", ...],
  "treatment_arms": [
    {"arm_name": "string", "dose": "string", "route": "string", "frequency": "string"}
  ],
  "visit_schedule": [
    {"visit_name": "string", "visit_day": "string", "assessments": ["string"]}
  ]
}

Rules:
- Return ONLY the JSON object — no preamble, no explanation.
- If a field is not found in the text, return an empty string or empty list.
- Be concise: list items should be one sentence maximum.
""".strip()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_pages: int = 40,
    ) -> None:
        self.model = model
        self.max_pages = max_pages
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key= to ProtocolParser()."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, pdf_path: str | Path) -> ProtocolMetadata:
        """
        Parse a protocol PDF and return structured ProtocolMetadata.

        Parameters
        ----------
        pdf_path : str or Path
            Path to the clinical protocol PDF.

        Returns
        -------
        ProtocolMetadata
            Dataclass containing all extracted study information.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Protocol PDF not found: {pdf_path}")

        logger.info("Extracting text from %s", pdf_path.name)
        raw_text = self._extract_text(pdf_path)

        logger.info("Sending %d characters to LLM (%s)", len(raw_text), self.model)
        extracted = self._call_llm(raw_text)

        metadata = self._build_metadata(extracted, raw_text_length=len(raw_text))
        logger.info("Parsing complete — %d endpoints extracted", len(metadata.primary_endpoints))
        return metadata

    def parse_to_json(self, pdf_path: str | Path, output_path: Optional[str | Path] = None) -> str:
        """
        Parse protocol and return (and optionally save) JSON output.

        Parameters
        ----------
        pdf_path : str or Path
            Path to the clinical protocol PDF.
        output_path : str or Path, optional
            If provided, saves the JSON to this file path.

        Returns
        -------
        str
            JSON string of extracted metadata.
        """
        metadata = self.parse(pdf_path)
        json_str = metadata.to_json()

        if output_path:
            Path(output_path).write_text(json_str, encoding="utf-8")
            logger.info("Saved output to %s", output_path)

        return json_str

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract plain text from PDF using PyMuPDF."""
        doc = fitz.open(str(pdf_path))
        pages = list(doc)[:self.max_pages]
        text = "\n\n".join(page.get_text() for page in pages)
        doc.close()
        return text.strip()

    def _call_llm(self, text: str) -> dict:
        """Send protocol text to LLM and return parsed JSON dict."""
        try:
            response = openai.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)

        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON: %s", exc)
            return {}
        except openai.OpenAIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise

    @staticmethod
    def _build_metadata(data: dict, raw_text_length: int = 0) -> ProtocolMetadata:
        """Map raw LLM dict output to ProtocolMetadata dataclass."""
        return ProtocolMetadata(
            study_title=data.get("study_title", ""),
            protocol_number=data.get("protocol_number", ""),
            phase=data.get("phase", ""),
            indication=data.get("indication", ""),
            sponsor=data.get("sponsor", ""),
            primary_endpoints=data.get("primary_endpoints", []),
            secondary_endpoints=data.get("secondary_endpoints", []),
            inclusion_criteria=data.get("inclusion_criteria", []),
            exclusion_criteria=data.get("exclusion_criteria", []),
            treatment_arms=data.get("treatment_arms", []),
            visit_schedule=data.get("visit_schedule", []),
            raw_text_length=raw_text_length,
        )
