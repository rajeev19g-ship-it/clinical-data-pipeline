"""
tests/test_tlf.py
──────────────────
Unit tests for TLF generation and CSR narrative drafting modules.

Uses synthetic ADaM datasets — no real patient data required.
LLM calls are mocked so tests run fully offline in CI/CD.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.tlf.generator import TLFGenerator, TLFOutput
from src.tlf.narrative import CSRNarrativeDrafter, NarrativeOutput


STUDY_ID = "DRUGX-2024-003"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def adsl_df() -> pd.DataFrame:
    """Synthetic ADSL dataset."""
    return pd.DataFrame({
        "USUBJID":  [f"{STUDY_ID}-01-00{i}" for i in range(1, 7)],
        "SITEID":   ["01", "01", "01", "02", "02", "02"],
        "AGE":      [45, 67, 38, 72, 55, 61],
        "AGEGR1":   ["<65", "65-74", "<65", ">=75", "<65", "<65"],
        "SEX":      ["M", "F", "M", "F", "M", "F"],
        "RACE":     ["WHITE", "ASIAN", "WHITE", "BLACK OR AFRICAN AMERICAN", "WHITE", "ASIAN"],
        "TRTA":     ["Drug X", "Placebo", "Drug X", "Placebo", "Drug X", "Placebo"],
        "TRT01A":   ["Drug X", "Placebo", "Drug X", "Placebo", "Drug X", "Placebo"],
        "TRT01AN":  [1, 2, 1, 2, 1, 2],
        "ITTFL":    ["Y"] * 6,
        "SAFFL":    ["Y"] * 6,
        "TRTSDT":   pd.to_datetime(["2024-01-17"] * 6),
    })


@pytest.fixture
def adae_df() -> pd.DataFrame:
    """Synthetic ADAE dataset."""
    return pd.DataFrame({
        "USUBJID":  [
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-002",
            f"{STUDY_ID}-02-004",
        ],
        "TRTA":     ["Drug X", "Drug X", "Placebo", "Placebo"],
        "AESEQ":    [1, 2, 1, 1],
        "AEDECOD":  ["NAUSEA", "FATIGUE", "HEADACHE", "RASH"],
        "AESEV":    ["MILD", "MODERATE", "MILD", "SEVERE"],
        "TRTEMFL":  ["Y", "Y", "Y", "Y"],
        "ASER":     ["N", "N", "N", "Y"],
        "AEREL":    ["Y", "N", "Y", "Y"],
        "ATOXGR":   [1, 2, 1, 3],
        "AEOUT":    ["RECOVERED/RESOLVED"] * 4,
    })


@pytest.fixture
def adlb_df() -> pd.DataFrame:
    """Synthetic ADLB dataset."""
    return pd.DataFrame({
        "USUBJID":  [
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-002",
            f"{STUDY_ID}-01-002",
        ],
        "TRTA":     ["Drug X", "Drug X", "Placebo", "Placebo"],
        "PARAMCD":  ["HGB", "HGB", "HGB", "HGB"],
        "PARAM":    ["Hemoglobin"] * 4,
        "AVAL":     [13.5, 11.2, 12.8, 10.5],
        "BASE":     [13.5, 13.5, 12.8, 12.8],
        "ABLFL":    ["Y", "", "Y", ""],
        "ANRIND":   ["NORMAL", "LOW", "NORMAL", "LOW"],
    })


@pytest.fixture
def adtte_df() -> pd.DataFrame:
    """Synthetic ADTTE dataset."""
    return pd.DataFrame({
        "USUBJID":  [f"{STUDY_ID}-01-00{i}" for i in range(1, 7)],
        "TRTA":     ["Drug X", "Placebo", "Drug X", "Placebo", "Drug X", "Placebo"],
        "TRT01AN":  [1, 2, 1, 2, 1, 2],
        "PARAMCD":  ["OS"] * 6,
        "PARAM":    ["Overall Survival"] * 6,
        "AVAL":     [245, 180, 310, 95, 280, 210],
        "AVALU":    ["DAYS"] * 6,
        "CNSR":     [0, 1, 0, 0, 1, 0],
        "EVNTDESC": ["DEATH", "CENSORED", "DEATH", "DEATH", "CENSORED", "DEATH"],
    })


@pytest.fixture
def generator() -> TLFGenerator:
    """TLFGenerator with Drug X / Placebo arms."""
    return TLFGenerator(
        study_id=STUDY_ID,
        treatment_var="TRTA",
        treatment_order=["Drug X", "Placebo"],
    )


# ── TLFOutput tests ───────────────────────────────────────────────────────────

class TestTLFOutput:

    def test_save_html_creates_file(self, tmp_path):
        output = TLFOutput(
            title="Test Table",
            table_number="14.1.1",
            content_html="<table><tr><td>Test</td></tr></table>",
            content_csv="col1,col2\nval1,val2",
            footnotes=["Note 1.", "Note 2."],
        )
        path = output.save_html(tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "14.1.1" in content
        assert "Test Table" in content

    def test_save_csv_creates_file(self, tmp_path):
        output = TLFOutput(
            title="Test Table",
            table_number="14.1.1",
            content_html="",
            content_csv="col1,col2\nval1,val2",
        )
        path = output.save_csv(tmp_path)
        assert path.exists()
        assert "col1" in path.read_text()

    def test_footnotes_in_html(self, tmp_path):
        output = TLFOutput(
            title="Test",
            table_number="99.1",
            content_html="",
            content_csv="",
            footnotes=["Safety population.", "TEAE definition."],
        )
        html = output._wrap_html()
        assert "Safety population." in html
        assert "TEAE definition." in html


# ── TLFGenerator — Demographics ───────────────────────────────────────────────

class TestDemographicsTable:

    def test_returns_tlf_output(self, generator, adsl_df):
        result = generator.demographics_table(adsl_df)
        assert isinstance(result, TLFOutput)

    def test_table_number(self, generator, adsl_df):
        result = generator.demographics_table(adsl_df)
        assert result.table_number == "14.1.1"

    def test_html_contains_age(self, generator, adsl_df):
        result = generator.demographics_table(adsl_df)
        assert "Age" in result.content_html

    def test_html_contains_sex(self, generator, adsl_df):
        result = generator.demographics_table(adsl_df)
        assert "Sex" in result.content_html

    def test_csv_parseable(self, generator, adsl_df):
        result = generator.demographics_table(adsl_df)
        import io
        df = pd.read_csv(io.StringIO(result.content_csv))
        assert len(df) > 0

    def test_footnotes_present(self, generator, adsl_df):
        result = generator.demographics_table(adsl_df)
        assert len(result.footnotes) > 0


# ── TLFGenerator — Adverse Events ────────────────────────────────────────────

class TestAdverseEventsTable:

    def test_returns_tlf_output(self, generator, adae_df, adsl_df):
        result = generator.adverse_events_table(adae_df, adsl_df)
        assert isinstance(result, TLFOutput)

    def test_table_number(self, generator, adae_df, adsl_df):
        result = generator.adverse_events_table(adae_df, adsl_df)
        assert result.table_number == "14.3.1"

    def test_html_contains_teae(self, generator, adae_df, adsl_df):
        result = generator.adverse_events_table(adae_df, adsl_df)
        assert "TEAE" in result.content_html

    def test_serious_row_present(self, generator, adae_df, adsl_df):
        result = generator.adverse_events_table(adae_df, adsl_df)
        assert "serious" in result.content_html.lower()

    def test_footnotes_present(self, generator, adae_df, adsl_df):
        result = generator.adverse_events_table(adae_df, adsl_df)
        assert any("TEAE" in fn for fn in result.footnotes)


# ── TLFGenerator — Lab Shift Table ───────────────────────────────────────────

class TestLabShiftTable:

    def test_returns_tlf_output(self, generator, adlb_df):
        result = generator.lab_shift_table(adlb_df, parameter="HGB")
        assert isinstance(result, TLFOutput)

    def test_table_number(self, generator, adlb_df):
        result = generator.lab_shift_table(adlb_df, parameter="HGB")
        assert result.table_number == "14.3.4"

    def test_title_contains_parameter(self, generator, adlb_df):
        result = generator.lab_shift_table(adlb_df, parameter="HGB")
        assert "HGB" in result.title

    def test_footnotes_present(self, generator, adlb_df):
        result = generator.lab_shift_table(adlb_df, parameter="HGB")
        assert len(result.footnotes) > 0


# ── TLFGenerator — Subject Listing ───────────────────────────────────────────

class TestSubjectListing:

    def test_returns_tlf_output(self, generator, adsl_df):
        result = generator.subject_listing(adsl_df)
        assert isinstance(result, TLFOutput)

    def test_table_number(self, generator, adsl_df):
        result = generator.subject_listing(adsl_df)
        assert result.table_number == "16.2.1"

    def test_all_subjects_present(self, generator, adsl_df):
        result = generator.subject_listing(adsl_df)
        import io
        df = pd.read_csv(io.StringIO(result.content_csv))
        assert len(df) == len(adsl_df)

    def test_custom_cols(self, generator, adsl_df):
        result = generator.subject_listing(adsl_df, cols=["USUBJID", "AGE", "TRTA"])
        import io
        df = pd.read_csv(io.StringIO(result.content_csv))
        assert "USUBJID" in df.columns
        assert "AGE" in df.columns


# ── CSRNarrativeDrafter tests ─────────────────────────────────────────────────

class TestCSRNarrativeDrafter:

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="OpenAI API key not found"):
                CSRNarrativeDrafter(api_key="")

    @patch("src.tlf.narrative.openai.chat.completions.create")
    def test_draft_demographics_returns_narrative(self, mock_openai, adsl_df):
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "A total of 6 subjects were enrolled and randomized, "
            "3 to Drug X and 3 to Placebo. The mean age was 56.3 years "
            "in the Drug X arm and 66.7 years in the Placebo arm."
        )
        mock_openai.return_value.choices = [mock_choice]

        drafter = CSRNarrativeDrafter(api_key="test-key")
        result = drafter.draft_demographics(
            adsl_df=adsl_df,
            study_id=STUDY_ID,
            indication="NSCLC",
            phase="Phase 3",
        )

        assert isinstance(result, NarrativeOutput)
        assert result.section == "11.2"
        assert len(result.narrative_text) > 0
        assert result.word_count > 0

    @patch("src.tlf.narrative.openai.chat.completions.create")
    def test_draft_safety_returns_narrative(self, mock_openai, adae_df, adsl_df):
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "Treatment-emergent adverse events were reported in 2 of 3 subjects "
            "in the Drug X arm and 1 of 3 subjects in the Placebo arm."
        )
        mock_openai.return_value.choices = [mock_choice]

        drafter = CSRNarrativeDrafter(api_key="test-key")
        result = drafter.draft_safety_summary(
            adae_df=adae_df,
            adsl_df=adsl_df,
            study_id=STUDY_ID,
        )

        assert isinstance(result, NarrativeOutput)
        assert result.section == "11.5"
        assert result.word_count > 0

    @patch("src.tlf.narrative.openai.chat.completions.create")
    def test_draft_efficacy_returns_narrative(self, mock_openai, adtte_df):
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "Overall survival events were observed in 2 of 3 subjects "
            "in the Drug X arm and 2 of 3 subjects in the Placebo arm."
        )
        mock_openai.return_value.choices = [mock_choice]

        drafter = CSRNarrativeDrafter(api_key="test-key")
        result = drafter.draft_efficacy_summary(
            adtte_df=adtte_df,
            study_id=STUDY_ID,
            primary_endpoint="OS",
        )

        assert isinstance(result, NarrativeOutput)
        assert result.section == "11.4"

    @patch("src.tlf.narrative.openai.chat.completions.create")
    def test_narrative_saves_to_file(self, mock_openai, adsl_df, tmp_path):
        mock_choice = MagicMock()
        mock_choice.message.content = "Six subjects were enrolled in this study."
        mock_openai.return_value.choices = [mock_choice]

        drafter = CSRNarrativeDrafter(api_key="test-key")
        result = drafter.draft_demographics(adsl_df, study_id=STUDY_ID)

        output_file = str(tmp_path / "section_11_2.txt")
        result.save(output_file)
        assert Path(output_file).exists()
        content = Path(output_file).read_text()
        assert "11.2" in content
