"""
tests/test_sdtm.py
───────────────────
Unit tests for SDTM domain mappers and define.xml generator.

Uses synthetic EDC-style source data — no real patient data required.
All tests run fully offline with no external dependencies.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.sdtm.domains import (
    AdverseEvents,
    Demographics,
    Exposure,
    LaboratoryResults,
    VitalSigns,
)
from src.sdtm.define_xml import DefineXMLGenerator


STUDY_ID = "DRUGX-2024-003"


# ── Synthetic source data fixtures ────────────────────────────────────────────

@pytest.fixture
def raw_dm() -> pd.DataFrame:
    """Synthetic raw demographics source data."""
    return pd.DataFrame({
        "subject_id":           ["001", "002", "003"],
        "site_id":              ["01",  "01",  "02"],
        "age":                  [45,     62,    38],
        "sex":                  ["Male", "Female", "Male"],
        "race":                 ["WHITE", "ASIAN", "BLACK OR AFRICAN AMERICAN"],
        "ethnicity":            ["NOT HISPANIC OR LATINO", "NOT HISPANIC OR LATINO", "HISPANIC OR LATINO"],
        "arm":                  ["Drug X", "Placebo", "Drug X"],
        "informed_consent_date":["2024-01-10", "2024-01-11", "2024-01-15"],
        "randomization_date":   ["2024-01-17", "2024-01-18", "2024-01-22"],
    })


@pytest.fixture
def raw_ae() -> pd.DataFrame:
    """Synthetic raw adverse events source data."""
    return pd.DataFrame({
        "subject_id":    ["001", "001", "002"],
        "site_id":       ["01",  "01",  "01"],
        "ae_term":       ["Nausea", "Fatigue", "Headache"],
        "ae_start_date": ["2024-02-01", "2024-02-05", "2024-02-03"],
        "ae_end_date":   ["2024-02-03", "2024-02-10", "2024-02-04"],
        "ae_severity":   ["Mild", "Moderate", "Mild"],
        "ae_serious":    ["No",   "No",        "No"],
        "ae_related":    ["Related", "Not Related", "Related"],
        "ae_outcome":    ["Recovered", "Recovered", "Resolved"],
    })


@pytest.fixture
def raw_lb() -> pd.DataFrame:
    """Synthetic raw laboratory results source data."""
    return pd.DataFrame({
        "subject_id":   ["001", "001", "002"],
        "site_id":      ["01",  "01",  "01"],
        "lab_test":     ["Hemoglobin", "Creatinine", "Hemoglobin"],
        "lab_result":   [13.5, 0.9, 11.2],
        "lab_unit":     ["g/dL", "mg/dL", "g/dL"],
        "lab_ref_low":  [12.0,   0.6,     12.0],
        "lab_ref_high": [17.0,   1.2,     17.0],
        "lab_date":     ["2024-01-17", "2024-01-17", "2024-01-18"],
        "visit_name":   ["Screening", "Screening", "Screening"],
    })


@pytest.fixture
def raw_vs() -> pd.DataFrame:
    """Synthetic raw vital signs source data."""
    return pd.DataFrame({
        "subject_id": ["001", "001", "002"],
        "site_id":    ["01",  "01",  "01"],
        "vs_test":    ["Systolic Blood Pressure", "Heart Rate", "Systolic Blood Pressure"],
        "vs_result":  [120,   72,    135],
        "vs_unit":    ["mmHg", "beats/min", "mmHg"],
        "vs_date":    ["2024-01-17", "2024-01-17", "2024-01-18"],
        "visit_name": ["Screening",  "Screening",  "Screening"],
    })


@pytest.fixture
def raw_ex() -> pd.DataFrame:
    """Synthetic raw exposure/dosing source data."""
    return pd.DataFrame({
        "subject_id":     ["001", "002"],
        "site_id":        ["01",  "01"],
        "drug_name":      ["Drug X",  "Placebo"],
        "dose_amount":    [200,       0],
        "dose_unit":      ["mg",      "mg"],
        "dose_route":     ["IV",      "IV"],
        "dose_frequency": ["Q3W",     "Q3W"],
        "start_date":     ["2024-01-17", "2024-01-18"],
        "end_date":       ["2024-06-17", "2024-06-18"],
        "visit_name":     ["Cycle 1 Day 1", "Cycle 1 Day 1"],
    })


# ── Demographics tests ─────────────────────────────────────────────────────────

class TestDemographics:

    def test_transform_returns_dataframe(self, raw_dm):
        dm = Demographics(study_id=STUDY_ID)
        result = dm.transform(raw_dm)
        assert isinstance(result, pd.DataFrame)

    def test_record_count(self, raw_dm):
        dm = Demographics(study_id=STUDY_ID)
        result = dm.transform(raw_dm)
        assert len(result) == 3

    def test_usubjid_format(self, raw_dm):
        dm = Demographics(study_id=STUDY_ID)
        result = dm.transform(raw_dm)
        assert all(result["USUBJID"].str.startswith(STUDY_ID))
        assert all(result["USUBJID"].str.count("-") >= 2)

    def test_sex_mapping(self, raw_dm):
        dm = Demographics(study_id=STUDY_ID)
        result = dm.transform(raw_dm)
        assert set(result["SEX"].unique()).issubset({"M", "F", "U"})

    def test_studyid_populated(self, raw_dm):
        dm = Demographics(study_id=STUDY_ID)
        result = dm.transform(raw_dm)
        assert all(result["STUDYID"] == STUDY_ID)

    def test_domain_is_dm(self, raw_dm):
        dm = Demographics(study_id=STUDY_ID)
        result = dm.transform(raw_dm)
        assert all(result["DOMAIN"] == "DM")

    def test_validation_passes(self, raw_dm):
        dm = Demographics(study_id=STUDY_ID)
        result = dm.transform(raw_dm)
        validation = dm.validate(result)
        assert validation.is_valid


# ── Adverse Events tests ───────────────────────────────────────────────────────

class TestAdverseEvents:

    def test_transform_returns_dataframe(self, raw_ae):
        ae = AdverseEvents(study_id=STUDY_ID)
        result = ae.transform(raw_ae)
        assert isinstance(result, pd.DataFrame)

    def test_severity_mapping(self, raw_ae):
        ae = AdverseEvents(study_id=STUDY_ID)
        result = ae.transform(raw_ae)
        assert "MILD" in result["AESEV"].values
        assert "MODERATE" in result["AESEV"].values

    def test_serious_flag(self, raw_ae):
        ae = AdverseEvents(study_id=STUDY_ID)
        result = ae.transform(raw_ae)
        assert set(result["AESER"].unique()).issubset({"Y", "N"})

    def test_sequence_numbers(self, raw_ae):
        ae = AdverseEvents(study_id=STUDY_ID)
        result = ae.transform(raw_ae)
        assert list(result["AESEQ"]) == [1, 2, 3]

    def test_outcome_mapping(self, raw_ae):
        ae = AdverseEvents(study_id=STUDY_ID)
        result = ae.transform(raw_ae)
        assert all(result["AEOUT"] == "RECOVERED/RESOLVED")

    def test_validation_passes(self, raw_ae):
        ae = AdverseEvents(study_id=STUDY_ID)
        result = ae.transform(raw_ae)
        validation = ae.validate(result)
        assert validation.is_valid


# ── Laboratory Results tests ───────────────────────────────────────────────────

class TestLaboratoryResults:

    def test_transform_returns_dataframe(self, raw_lb):
        lb = LaboratoryResults(study_id=STUDY_ID)
        result = lb.transform(raw_lb)
        assert isinstance(result, pd.DataFrame)

    def test_nrind_derivation(self, raw_lb):
        lb = LaboratoryResults(study_id=STUDY_ID)
        result = lb.transform(raw_lb)
        assert set(result["LBNRIND"].unique()).issubset({"NORMAL", "LOW", "HIGH", "UNKNOWN"})

    def test_low_hemoglobin_flagged(self, raw_lb):
        lb = LaboratoryResults(study_id=STUDY_ID)
        result = lb.transform(raw_lb)
        hgb_low = result[
            (result["LBTEST"] == "HEMOGLOBIN") &
            (result["LBSTRESN"] < 12.0)
        ]
        assert all(hgb_low["LBNRIND"] == "LOW")

    def test_testcd_max_8_chars(self, raw_lb):
        lb = LaboratoryResults(study_id=STUDY_ID)
        result = lb.transform(raw_lb)
        assert all(result["LBTESTCD"].str.len() <= 8)

    def test_validation_passes(self, raw_lb):
        lb = LaboratoryResults(study_id=STUDY_ID)
        result = lb.transform(raw_lb)
        validation = lb.validate(result)
        assert validation.is_valid


# ── Vital Signs tests ──────────────────────────────────────────────────────────

class TestVitalSigns:

    def test_transform_returns_dataframe(self, raw_vs):
        vs = VitalSigns(study_id=STUDY_ID)
        result = vs.transform(raw_vs)
        assert isinstance(result, pd.DataFrame)

    def test_testcd_mapping(self, raw_vs):
        vs = VitalSigns(study_id=STUDY_ID)
        result = vs.transform(raw_vs)
        assert "SYSBP" in result["VSTESTCD"].values
        assert "HR" in result["VSTESTCD"].values

    def test_numeric_results(self, raw_vs):
        vs = VitalSigns(study_id=STUDY_ID)
        result = vs.transform(raw_vs)
        assert pd.api.types.is_numeric_dtype(result["VSSTRESN"])

    def test_validation_passes(self, raw_vs):
        vs = VitalSigns(study_id=STUDY_ID)
        result = vs.transform(raw_vs)
        validation = vs.validate(result)
        assert validation.is_valid


# ── Exposure tests ─────────────────────────────────────────────────────────────

class TestExposure:

    def test_transform_returns_dataframe(self, raw_ex):
        ex = Exposure(study_id=STUDY_ID)
        result = ex.transform(raw_ex)
        assert isinstance(result, pd.DataFrame)

    def test_drug_name_uppercased(self, raw_ex):
        ex = Exposure(study_id=STUDY_ID)
        result = ex.transform(raw_ex)
        assert all(result["EXTRT"] == result["EXTRT"].str.upper())

    def test_dose_numeric(self, raw_ex):
        ex = Exposure(study_id=STUDY_ID)
        result = ex.transform(raw_ex)
        assert pd.api.types.is_numeric_dtype(result["EXDOSE"])

    def test_validation_passes(self, raw_ex):
        ex = Exposure(study_id=STUDY_ID)
        result = ex.transform(raw_ex)
        validation = ex.validate(result)
        assert validation.is_valid


# ── DefineXMLGenerator tests ───────────────────────────────────────────────────

class TestDefineXMLGenerator:

    def test_write_creates_file(self, raw_dm, raw_ae, tmp_path):
        dm = Demographics(study_id=STUDY_ID).transform(raw_dm)
        ae = AdverseEvents(study_id=STUDY_ID).transform(raw_ae)

        gen = DefineXMLGenerator(
            study_id=STUDY_ID,
            study_description="Phase 3 study of Drug X in NSCLC",
        )
        gen.add_domain("DM", dm)
        gen.add_domain("AE", ae)

        output = tmp_path / "define.xml"
        gen.write(output)
        assert output.exists()

    def test_xml_contains_study_id(self, raw_dm, tmp_path):
        dm = Demographics(study_id=STUDY_ID).transform(raw_dm)
        gen = DefineXMLGenerator(study_id=STUDY_ID)
        gen.add_domain("DM", dm)
        output = tmp_path / "define.xml"
        gen.write(output)
        content = output.read_text()
        assert STUDY_ID in content

    def test_summary_keys(self, raw_dm):
        dm = Demographics(study_id=STUDY_ID).transform(raw_dm)
        gen = DefineXMLGenerator(study_id=STUDY_ID)
        gen.add_domain("DM", dm)
        summary = gen.summary()
        assert "total_datasets" in summary
        assert summary["total_datasets"] == 1

    def test_multiple_domains(self, raw_dm, raw_ae, raw_lb):
        dm = Demographics(study_id=STUDY_ID).transform(raw_dm)
        ae = AdverseEvents(study_id=STUDY_ID).transform(raw_ae)
        lb = LaboratoryResults(study_id=STUDY_ID).transform(raw_lb)

        gen = DefineXMLGenerator(study_id=STUDY_ID)
        gen.add_domain("DM", dm)
        gen.add_domain("AE", ae)
        gen.add_domain("LB", lb)

        assert gen.summary()["total_datasets"] == 3
