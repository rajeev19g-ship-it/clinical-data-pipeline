Perfect! That's exactly right! 🎉

✅ adam
✅ models
✅ protocol_automation
✅ sdtm — just created with base.py inside!
✅ tlf

Now let's add domains.py — the actual DM, AE, LB, VS, EX mappers:

Click the sdtm folder
Click "Add file" → "Create new file"
Type in the filename box:

domains.py

Paste this code:

python"""
sdtm/domains.py
────────────────
Concrete SDTM domain mappers for the most common submission domains.

Domains implemented:
    DM  - Demographics
    AE  - Adverse Events
    LB  - Laboratory Test Results
    VS  - Vital Signs
    EX  - Exposure

Each class inherits from SDTMDomain and implements transform()
to convert raw EDC data into CDISC SDTM 3.3 compliant datasets.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import SDTMDomain

logger = logging.getLogger(__name__)


# ── DM - Demographics ─────────────────────────────────────────────────────────

class Demographics(SDTMDomain):
    """
    SDTM DM domain mapper.

    Maps raw subject demographics to CDISC DM domain.
    DM is the anchor dataset — every subject in the study
    must appear here exactly once.

    Required source columns
    -----------------------
    subject_id, site_id, age, sex, race, ethnicity,
    informed_consent_date, randomization_date, arm
    """

    DOMAIN = "DM"
    REQUIRED_VARS = [
        "STUDYID", "DOMAIN", "USUBJID", "SUBJID", "SITEID",
        "AGE", "AGEU", "SEX", "RACE", "ETHNIC",
        "ARMCD", "ARM", "RFSTDTC", "RFICDTC",
    ]

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        dm = pd.DataFrame()
        dm["SUBJID"]  = raw_df["subject_id"].astype(str)
        dm["SITEID"]  = raw_df["site_id"].astype(str)
        dm["USUBJID"] = (
            self.study_id + "-"
            + dm["SITEID"] + "-"
            + dm["SUBJID"]
        )
        dm["AGE"]     = pd.to_numeric(raw_df["age"], errors="coerce")
        dm["AGEU"]    = "YEARS"
        dm["SEX"]     = raw_df["sex"].str.upper().map(
            {"M": "M", "F": "F", "MALE": "M", "FEMALE": "F", "U": "U"}
        ).fillna("U")
        dm["RACE"]    = raw_df["race"].str.upper()
        dm["ETHNIC"]  = raw_df.get("ethnicity", pd.Series("UNKNOWN", index=raw_df.index)).str.upper()
        dm["ARMCD"]   = raw_df["arm"].str.upper().str.replace(" ", "_")
        dm["ARM"]     = raw_df["arm"]
        dm["RFICDTC"] = pd.to_datetime(
            raw_df["informed_consent_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        dm["RFSTDTC"] = pd.to_datetime(
            raw_df["randomization_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        dm = self._add_standard_vars(dm)
        logger.info("DM: transformed %d subjects", len(dm))
        return dm


# ── AE - Adverse Events ───────────────────────────────────────────────────────

class AdverseEvents(SDTMDomain):
    """
    SDTM AE domain mapper.

    Maps raw adverse event data to CDISC AE domain.
    Supports MedDRA coding fields, CTCAE grading,
    and serious/related flags.

    Required source columns
    -----------------------
    subject_id, site_id, ae_term, ae_start_date,
    ae_end_date, ae_severity, ae_serious, ae_related,
    ae_outcome
    """

    DOMAIN = "AE"
    REQUIRED_VARS = [
        "STUDYID", "DOMAIN", "USUBJID", "AESEQ",
        "AETERM", "AEDECOD", "AESTDTC", "AEENDTC",
        "AESEV", "AESER", "AEREL", "AEOUT",
    ]

    _SEVERITY_MAP = {
        "MILD": "MILD", "1": "MILD", "GRADE 1": "MILD",
        "MODERATE": "MODERATE", "2": "MODERATE", "GRADE 2": "MODERATE",
        "SEVERE": "SEVERE", "3": "SEVERE", "GRADE 3": "SEVERE",
        "LIFE-THREATENING": "LIFE-THREATENING", "4": "LIFE-THREATENING",
        "FATAL": "FATAL", "5": "FATAL",
    }

    _OUTCOME_MAP = {
        "RECOVERED": "RECOVERED/RESOLVED",
        "RESOLVED": "RECOVERED/RESOLVED",
        "RECOVERING": "RECOVERING/RESOLVING",
        "NOT RECOVERED": "NOT RECOVERED/NOT RESOLVED",
        "FATAL": "FATAL",
        "UNKNOWN": "UNKNOWN",
    }

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        ae = pd.DataFrame()
        ae["USUBJID"]  = (
            self.study_id + "-"
            + raw_df["site_id"].astype(str) + "-"
            + raw_df["subject_id"].astype(str)
        )
        ae["AETERM"]   = raw_df["ae_term"].str.upper()
        ae["AEDECOD"]  = raw_df.get("ae_decoded_term", raw_df["ae_term"]).str.upper()
        ae["AEBODSYS"] = raw_df.get("ae_body_system", pd.Series("", index=raw_df.index)).str.upper()
        ae["AESTDTC"]  = pd.to_datetime(
            raw_df["ae_start_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        ae["AEENDTC"]  = pd.to_datetime(
            raw_df["ae_end_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        ae["AESEV"]    = (
            raw_df["ae_severity"].str.upper()
            .map(self._SEVERITY_MAP)
            .fillna("UNKNOWN")
        )
        ae["AESER"]    = raw_df["ae_serious"].str.upper().map(
            {"YES": "Y", "NO": "N", "Y": "Y", "N": "N"}
        ).fillna("N")
        ae["AEREL"]    = raw_df["ae_related"].str.upper().map(
            {"RELATED": "Y", "NOT RELATED": "N", "YES": "Y", "NO": "N"}
        ).fillna("N")
        ae["AEOUT"]    = (
            raw_df["ae_outcome"].str.upper()
            .map(self._OUTCOME_MAP)
            .fillna("UNKNOWN")
        )
        ae["AESEQ"]    = range(1, len(ae) + 1)

        ae = self._add_standard_vars(ae)
        logger.info("AE: transformed %d adverse event records", len(ae))
        return ae


# ── LB - Laboratory Test Results ─────────────────────────────────────────────

class LaboratoryResults(SDTMDomain):
    """
    SDTM LB domain mapper.

    Maps raw lab data to CDISC LB domain including
    reference range flags and toxicity grade derivation.

    Required source columns
    -----------------------
    subject_id, site_id, lab_test, lab_result,
    lab_unit, lab_ref_low, lab_ref_high,
    lab_date, visit_name
    """

    DOMAIN = "LB"
    REQUIRED_VARS = [
        "STUDYID", "DOMAIN", "USUBJID", "LBSEQ",
        "LBTESTCD", "LBTEST", "LBCAT",
        "LBORRES", "LBORRESU", "LBSTRESN",
        "LBSTNRLO", "LBSTNRHI", "LBNRIND",
        "LBDTC", "VISIT",
    ]

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        lb = pd.DataFrame()
        lb["USUBJID"]  = (
            self.study_id + "-"
            + raw_df["site_id"].astype(str) + "-"
            + raw_df["subject_id"].astype(str)
        )
        lb["LBTESTCD"] = raw_df["lab_test"].str.upper().str.replace(" ", "").str[:8]
        lb["LBTEST"]   = raw_df["lab_test"].str.upper()
        lb["LBCAT"]    = raw_df.get("lab_category", pd.Series("CHEMISTRY", index=raw_df.index)).str.upper()
        lb["LBORRES"]  = raw_df["lab_result"].astype(str)
        lb["LBORRESU"] = raw_df["lab_unit"].str.upper()
        lb["LBSTRESN"] = pd.to_numeric(raw_df["lab_result"], errors="coerce")
        lb["LBSTRESU"] = raw_df["lab_unit"].str.upper()
        lb["LBSTNRLO"] = pd.to_numeric(raw_df["lab_ref_low"], errors="coerce")
        lb["LBSTNRHI"] = pd.to_numeric(raw_df["lab_ref_high"], errors="coerce")
        lb["LBNRIND"]  = self._derive_nrind(lb)
        lb["LBDTC"]    = pd.to_datetime(
            raw_df["lab_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        lb["VISIT"]    = raw_df["visit_name"].str.upper()
        lb["LBSEQ"]    = range(1, len(lb) + 1)

        lb = self._add_standard_vars(lb)
        logger.info("LB: transformed %d lab records", len(lb))
        return lb

    @staticmethod
    def _derive_nrind(lb: pd.DataFrame) -> pd.Series:
        """Derive normal range indicator (LBNRIND) from result vs reference range."""
        conditions = [
            lb["LBSTRESN"] < lb["LBSTNRLO"],
            lb["LBSTRESN"] > lb["LBSTNRHI"],
            lb["LBSTRESN"].between(lb["LBSTNRLO"], lb["LBSTNRHI"]),
        ]
        choices = ["LOW", "HIGH", "NORMAL"]
        return np.select(conditions, choices, default="UNKNOWN")


# ── VS - Vital Signs ──────────────────────────────────────────────────────────

class VitalSigns(SDTMDomain):
    """
    SDTM VS domain mapper.

    Maps raw vital signs data to CDISC VS domain.

    Required source columns
    -----------------------
    subject_id, site_id, vs_test, vs_result,
    vs_unit, vs_date, visit_name, vs_position
    """

    DOMAIN = "VS"
    REQUIRED_VARS = [
        "STUDYID", "DOMAIN", "USUBJID", "VSSEQ",
        "VSTESTCD", "VSTEST", "VSORRES", "VSORRESU",
        "VSSTRESN", "VSSTRESU", "VSDTC", "VISIT",
    ]

    _TESTCD_MAP = {
        "SYSTOLIC BLOOD PRESSURE": "SYSBP",
        "DIASTOLIC BLOOD PRESSURE": "DIABP",
        "HEART RATE": "HR",
        "BODY TEMPERATURE": "TEMP",
        "RESPIRATORY RATE": "RESP",
        "WEIGHT": "WEIGHT",
        "HEIGHT": "HEIGHT",
        "BMI": "BMI",
        "OXYGEN SATURATION": "OXYSAT",
    }

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        vs = pd.DataFrame()
        vs["USUBJID"]  = (
            self.study_id + "-"
            + raw_df["site_id"].astype(str) + "-"
            + raw_df["subject_id"].astype(str)
        )
        vs["VSTESTCD"] = (
            raw_df["vs_test"].str.upper()
            .map(self._TESTCD_MAP)
            .fillna(raw_df["vs_test"].str.upper().str[:8])
        )
        vs["VSTEST"]   = raw_df["vs_test"].str.upper()
        vs["VSPOS"]    = raw_df.get("vs_position", pd.Series("SITTING", index=raw_df.index)).str.upper()
        vs["VSORRES"]  = raw_df["vs_result"].astype(str)
        vs["VSORRESU"] = raw_df["vs_unit"].str.upper()
        vs["VSSTRESN"] = pd.to_numeric(raw_df["vs_result"], errors="coerce")
        vs["VSSTRESU"] = raw_df["vs_unit"].str.upper()
        vs["VSDTC"]    = pd.to_datetime(
            raw_df["vs_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        vs["VISIT"]    = raw_df["visit_name"].str.upper()
        vs["VSSEQ"]    = range(1, len(vs) + 1)

        vs = self._add_standard_vars(vs)
        logger.info("VS: transformed %d vital sign records", len(vs))
        return vs


# ── EX - Exposure ─────────────────────────────────────────────────────────────

class Exposure(SDTMDomain):
    """
    SDTM EX domain mapper.

    Maps raw drug administration / dosing records
    to CDISC EX domain.

    Required source columns
    -----------------------
    subject_id, site_id, drug_name, dose_amount,
    dose_unit, dose_route, dose_frequency,
    start_date, end_date, visit_name
    """

    DOMAIN = "EX"
    REQUIRED_VARS = [
        "STUDYID", "DOMAIN", "USUBJID", "EXSEQ",
        "EXTRT", "EXDOSE", "EXDOSU", "EXROUTE",
        "EXDOSFRQ", "EXSTDTC", "EXENDTC", "VISIT",
    ]

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        ex = pd.DataFrame()
        ex["USUBJID"]   = (
            self.study_id + "-"
            + raw_df["site_id"].astype(str) + "-"
            + raw_df["subject_id"].astype(str)
        )
        ex["EXTRT"]     = raw_df["drug_name"].str.upper()
        ex["EXDOSE"]    = pd.to_numeric(raw_df["dose_amount"], errors="coerce")
        ex["EXDOSU"]    = raw_df["dose_unit"].str.upper()
        ex["EXROUTE"]   = raw_df["dose_route"].str.upper()
        ex["EXDOSFRQ"]  = raw_df["dose_frequency"].str.upper()
        ex["EXSTDTC"]   = pd.to_datetime(
            raw_df["start_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        ex["EXENDTC"]   = pd.to_datetime(
            raw_df["end_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        ex["VISIT"]     = raw_df["visit_name"].str.upper()
        ex["EXSEQ"]     = range(1, len(ex) + 1)

        ex = self._add_standard_vars(ex)
        logger.info("EX: transformed %d exposure records", len(ex))
        return ex
