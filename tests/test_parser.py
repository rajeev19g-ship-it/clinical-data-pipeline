"""
tests/test_parser.py
─────────────────────
Unit tests for protocol_automation.parser and variable_mapper modules.

Uses synthetic protocol data — no real patient data or API calls required.
LLM calls are mocked so tests run fully offline in CI/CD.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.protocol_automation.parser import ProtocolMetadata, ProtocolParser
from src.protocol_automation.variable_mapper import MappingTable, VariableMapper


# ── Fixtures ──────────────────────────────────────────────────────────────────

SYNTHETIC_LLM_RESPONSE = {
    "study_title": "A Phase 3 Randomized Controlled Trial of Drug X in Adult Patients with NSCLC",
    "protocol_number": "DRUGX-2024-003",
    "phase": "Phase 3",
    "indication": "Non-Small Cell Lung Cancer (NSCLC)",
    "sponsor": "Synthetic Pharma Inc.",
    "primary_endpoints": [
        "Overall survival (OS) defined as time from randomization to death from any cause",
        "Progression-free survival (PFS) per RECIST 1.1 assessed by blinded independent review",
    ],
    "secondary_endpoints": [
        "Objective response rate (ORR) per RECIST 1.1",
        "Duration of response (DoR)",
        "Patient reported outcomes via EORTC QLQ-C30 questionnaire",
        "Safety and tolerability including adverse events per CTCAE v5.0",
    ],
    "inclusion_criteria": [
        "Age >= 18 years at time of informed consent",
        "Histologically confirmed NSCLC with stage IIIB or IV disease",
        "ECOG performance status 0 or 1",
        "Adequate hematology laboratory values: ANC >= 1.5 x 10^9/L",
    ],
    "exclusion_criteria": [
        "Prior treatment with any anti-PD-1 or anti-PD-L1 antibody",
        "Active or prior documented autoimmune disease",
        "Known active central nervous system metastases",
    ],
    "treatment_arms": [
        {"arm_name": "Drug X", "dose": "200mg", "route": "IV", "frequency": "Q3W"},
        {"arm_name": "Placebo", "dose": "N/A", "route": "IV", "frequency": "Q3W"},
    ],
    "visit_schedule": [
        {
            "visit_name": "Screening",
            "visit_day": "Day -28 to Day -1",
            "assessments": ["Demographics", "Medical history", "Laboratory tests", "ECG"],
        },
        {
            "visit_name": "Cycle 1 Day 1",
            "visit_day": "Day 1",
            "assessments": ["Vital signs", "ECOG", "Drug administration", "PK sample"],
        },
    ],
}


@pytest.fixture
def synthetic_metadata() -> ProtocolMetadata:
    """Return a ProtocolMetadata object built from synthetic LLM response."""
    return ProtocolParser._build_metadata(SYNTHETIC_LLM_RESPONSE, raw_text_length=15000)


@pytest.fixture
def mapper() -> VariableMapper:
    """Return a default VariableMapper instance."""
    return VariableMapper()


# ── ProtocolMetadata tests ─────────────────────────────────────────────────────

class TestProtocolMetadata:

    def test_to_dict_keys(self, synthetic_metadata):
        """All expected keys are present in serialized dict."""
        result = synthetic_metadata.to_dict()
        expected_keys = {
            "study_title", "protocol_number", "phase", "indication",
            "sponsor", "primary_endpoints", "secondary_endpoints",
            "inclusion_criteria", "exclusion_criteria",
            "treatment_arms", "visit_schedule", "raw_text_length",
        }
        assert expected_keys == set(result.keys())

    def test_to_json_valid(self, synthetic_metadata):
        """to_json() returns valid JSON string."""
        json_str = synthetic_metadata.to_json()
        parsed = json.loads(json_str)
        assert parsed["protocol_number"] == "DRUGX-2024-003"

    def test_phase_extracted(self, synthetic_metadata):
        """Phase is correctly extracted."""
        assert synthetic_metadata.phase == "Phase 3"

    def test_primary_endpoints_count(self, synthetic_metadata):
        """Correct number of primary endpoints extracted."""
        assert len(synthetic_metadata.primary_endpoints) == 2

    def test_treatment_arms_structure(self, synthetic_metadata):
        """Treatment arms contain required keys."""
        for arm in synthetic_metadata.treatment_arms:
            assert "arm_name" in arm
            assert "dose" in arm
            assert "route" in arm

    def test_raw_text_length(self, synthetic_metadata):
        """Raw text length is recorded."""
        assert synthetic_metadata.raw_text_length == 15000


# ── ProtocolParser tests ───────────────────────────────────────────────────────

class TestProtocolParser:

    def test_missing_api_key_raises(self):
        """ProtocolParser raises EnvironmentError when no API key is set."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="OpenAI API key not found"):
                ProtocolParser(api_key="")

    def test_missing_pdf_raises(self):
        """parse() raises FileNotFoundError for non-existent PDF."""
        parser = ProtocolParser(api_key="test-key-123")
        with pytest.raises(FileNotFoundError):
            parser.parse("does_not_exist.pdf")

    @patch("src.protocol_automation.parser.openai.chat.completions.create")
    @patch("src.protocol_automation.parser.fitz.open")
    def test_parse_returns_metadata(self, mock_fitz, mock_openai, tmp_path):
        """parse() returns ProtocolMetadata when PDF exists and LLM responds."""
        # Mock PDF text extraction
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Synthetic protocol text content."
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_fitz.return_value = mock_doc

        # Mock LLM response
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(SYNTHETIC_LLM_RESPONSE)
        mock_openai.return_value.choices = [mock_choice]

        # Create a dummy PDF file
        dummy_pdf = tmp_path / "protocol.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 synthetic")

        parser = ProtocolParser(api_key="test-key-123")
        metadata = parser.parse(dummy_pdf)

        assert isinstance(metadata, ProtocolMetadata)
        assert metadata.study_title == SYNTHETIC_LLM_RESPONSE["study_title"]
        assert metadata.phase == "Phase 3"

    @patch("src.protocol_automation.parser.openai.chat.completions.create")
    @patch("src.protocol_automation.parser.fitz.open")
    def test_parse_to_json_saves_file(self, mock_fitz, mock_openai, tmp_path):
        """parse_to_json() saves JSON file to specified output path."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Synthetic protocol text."
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_fitz.return_value = mock_doc

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(SYNTHETIC_LLM_RESPONSE)
        mock_openai.return_value.choices = [mock_choice]

        dummy_pdf = tmp_path / "protocol.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 synthetic")

        output_path = tmp_path / "output.json"
        parser = ProtocolParser(api_key="test-key-123")
        parser.parse_to_json(dummy_pdf, output_path=output_path)

        assert output_path.exists()
        saved = json.loads(output_path.read_text())
        assert saved["protocol_number"] == "DRUGX-2024-003"


# ── VariableMapper tests ───────────────────────────────────────────────────────

class TestVariableMapper:

    def test_returns_mapping_table(self, mapper, synthetic_metadata):
        """map() returns a MappingTable instance."""
        table = mapper.map(synthetic_metadata)
        assert isinstance(table, MappingTable)

    def test_mapping_count(self, mapper, synthetic_metadata):
        """Total mappings equals sum of all endpoint/criteria lists."""
        table = mapper.map(synthetic_metadata)
        expected = (
            len(synthetic_metadata.primary_endpoints)
            + len(synthetic_metadata.secondary_endpoints)
            + len(synthetic_metadata.inclusion_criteria)
            + len(synthetic_metadata.exclusion_criteria)
        )
        assert len(table.mappings) == expected

    def test_survival_maps_to_adtte(self, mapper, synthetic_metadata):
        """Overall survival endpoint maps to ADTTE."""
        table = mapper.map(synthetic_metadata)
        os_mapping = next(
            m for m in table.mappings
            if "overall survival" in m.endpoint.lower()
        )
        assert os_mapping.suggested_adam_dataset == "ADTTE"

    def test_lab_maps_to_lb(self, mapper, synthetic_metadata):
        """Laboratory inclusion criterion maps to LB domain."""
        table = mapper.map(synthetic_metadata)
        lab_mapping = next(
            m for m in table.mappings
            if "laboratory" in m.endpoint.lower()
        )
        assert lab_mapping.suggested_sdtm_domain == "LB"

    def test_csv_export(self, mapper, synthetic_metadata):
        """to_csv() produces a non-empty CSV string with header row."""
        table = mapper.map(synthetic_metadata)
        csv_str = table.to_csv()
        lines = csv_str.strip().split("\n")
        assert lines[0].startswith("endpoint")
        assert len(lines) > 1

    def test_summary_keys(self, mapper, synthetic_metadata):
        """summary() contains all expected keys."""
        table = mapper.map(synthetic_metadata)
        summary = table.summary()
        assert "total_endpoints" in summary
        assert "unique_sdtm_domains" in summary
        assert "suppqual_required" in summary
        assert "low_confidence_count" in summary

    def test_unknown_endpoint_confidence_low(self, mapper):
        """Completely unrecognizable endpoint gets low confidence."""
        from src.protocol_automation.parser import ProtocolMetadata
        dummy = ProtocolMetadata(
            primary_endpoints=["xyzzy frobnicate quantum entanglement score"],
        )
        table = mapper.map(dummy)
        assert table.mappings[0].confidence == "low"
        assert table.mappings[0].requires_suppqual is True
