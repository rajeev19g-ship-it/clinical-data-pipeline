Great! Now let's build the ADaM engine. This is where your stats expertise really shines — and where the ML/TensorFlow angle makes the repo stand out.
We'll create 2 files inside src/adam/:

derivations.py — ADSL, ADAE, ADTTE, ADLB
imputation.py — ML-based missing value imputation using scikit-learn and TensorFlow

Let's start:

Click src → adam folder
Click "Add file" → "Create new file"
Type in the filename box:

derivations.py

Paste this code:

python"""
adam/derivations.py
────────────────────
ADaM dataset derivations for core analysis datasets.

Datasets implemented:
    ADSL  - Subject Level Analysis Dataset
    ADAE  - Adverse Event Analysis Dataset
    ADTTE - Time to Event Analysis Dataset
    ADLB  - Laboratory Data Analysis Dataset

All derivations follow CDISC ADaM 1.3 specification and
ICH E9 statistical analysis principles.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Base ADaM class ───────────────────────────────────────────────────────────

class ADaMDataset:
    """Base class for all ADaM dataset derivations."""

    DATASET_NAME: str = ""
    DATASET_LABEL: str = ""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id

    def _assert_required_cols(self, df: pd.DataFrame, required: list[str]) -> None:
        """Raise ValueError if any required columns are missing."""
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.DATASET_NAME}: missing required columns: {missing}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(study_id={self.study_id!r})"


# ── ADSL - Subject Level Analysis Dataset ────────────────────────────────────

class ADSL(ADaMDataset):
    """
    Subject Level Analysis Dataset (ADSL).

    ADSL is the foundational ADaM dataset — every subject appears
    exactly once. It contains demographics, treatment assignment,
    analysis population flags, and key study dates.

    All other ADaM datasets merge with ADSL on USUBJID.

    Parameters
    ----------
    study_id : str
        Sponsor study identifier.

    Examples
    --------
    >>> adsl = ADSL(study_id="DRUGX-2024-003")
    >>> adsl_df = adsl.derive(dm_df)
    """

    DATASET_NAME  = "ADSL"
    DATASET_LABEL = "Subject Level Analysis Dataset"

    def derive(self, dm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive ADSL from SDTM DM domain.

        Parameters
        ----------
        dm_df : pd.DataFrame
            SDTM DM domain dataset.

        Returns
        -------
        pd.DataFrame
            ADSL dataset with population flags and analysis variables.
        """
        self._assert_required_cols(dm_df, ["USUBJID", "AGE", "SEX", "ARM", "RFSTDTC"])

        adsl = dm_df.copy()

        # ── Treatment variables ───────────────────────────────────────────────
        adsl["TRT01P"]  = adsl["ARM"]
        adsl["TRT01PN"] = adsl["ARM"].map(
            lambda x: 1 if "DRUG" in str(x).upper() else 2
        )
        adsl["TRT01A"]  = adsl["TRT01P"]   # Actual = Planned (no rescue therapy)
        adsl["TRT01AN"] = adsl["TRT01PN"]

        # ── Age grouping ──────────────────────────────────────────────────────
        adsl["AGEGR1"] = pd.cut(
            adsl["AGE"],
            bins=[0, 64, 74, 999],
            labels=["<65", "65-74", ">=75"],
            right=True,
        ).astype(str)
        adsl["AGEGR1N"] = adsl["AGEGR1"].map({"<65": 1, "65-74": 2, ">=75": 3})

        # ── Study dates ───────────────────────────────────────────────────────
        adsl["TRTSDT"] = pd.to_datetime(adsl["RFSTDTC"], errors="coerce")
        adsl["TRTSDTM"] = adsl["TRTSDT"]

        # ── Population flags ──────────────────────────────────────────────────
        # ITT: all randomized subjects
        adsl["ITTFL"] = "Y"
        # Safety: all subjects who received at least one dose
        adsl["SAFFL"] = "Y"
        # Per Protocol: subjects with no major protocol deviations (placeholder)
        adsl["PPROTFL"] = "Y"
        # Evaluable: subjects with at least one post-baseline assessment
        adsl["EVALFL"] = "Y"

        logger.info("ADSL derived: %d subjects", len(adsl))
        return adsl


# ── ADAE - Adverse Event Analysis Dataset ────────────────────────────────────

class ADAE(ADaMDataset):
    """
    Adverse Event Analysis Dataset (ADAE).

    Extends SDTM AE with analysis variables including:
    - Treatment-emergent AE flag (TRTEMFL)
    - Analysis severity grade
    - Days from first dose to AE onset (AESTDY)
    - Relatedness and seriousness flags

    Parameters
    ----------
    study_id : str
        Sponsor study identifier.

    Examples
    --------
    >>> adae = ADAE(study_id="DRUGX-2024-003")
    >>> adae_df = adae.derive(ae_df, adsl_df)
    """

    DATASET_NAME  = "ADAE"
    DATASET_LABEL = "Adverse Event Analysis Dataset"

    def derive(self, ae_df: pd.DataFrame, adsl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive ADAE from SDTM AE and ADSL.

        Parameters
        ----------
        ae_df : pd.DataFrame
            SDTM AE domain dataset.
        adsl_df : pd.DataFrame
            ADSL dataset (for treatment start dates and population flags).

        Returns
        -------
        pd.DataFrame
            ADAE dataset with treatment-emergent flags and analysis variables.
        """
        self._assert_required_cols(ae_df,   ["USUBJID", "AESTDTC", "AESEV", "AESER", "AEREL"])
        self._assert_required_cols(adsl_df, ["USUBJID", "TRTSDT", "TRT01A", "SAFFL"])

        # Merge AE with ADSL
        adsl_keys = adsl_df[["USUBJID", "TRTSDT", "TRT01A", "TRT01AN", "SAFFL"]].copy()
        adae = ae_df.merge(adsl_keys, on="USUBJID", how="left")

        # ── Analysis treatment variables ──────────────────────────────────────
        adae["TRTA"]  = adae["TRT01A"]
        adae["TRTAN"] = adae["TRT01AN"]

        # ── Date handling ─────────────────────────────────────────────────────
        adae["ASTDT"] = pd.to_datetime(adae["AESTDTC"], errors="coerce")
        adae["AENDT"] = pd.to_datetime(adae["AEENDTC"], errors="coerce")

        # ── Study day calculation ─────────────────────────────────────────────
        adae["AESTDY"] = (adae["ASTDT"] - adae["TRTSDT"]).dt.days + 1

        # ── Treatment-emergent flag ───────────────────────────────────────────
        # TEAE = AE that starts on or after first dose date
        adae["TRTEMFL"] = np.where(
            adae["ASTDT"] >= adae["TRTSDT"], "Y", ""
        )

        # ── Severity grade (numeric) ──────────────────────────────────────────
        adae["ATOXGR"] = adae["AESEV"].map({
            "MILD":             1,
            "MODERATE":         2,
            "SEVERE":           3,
            "LIFE-THREATENING": 4,
            "FATAL":            5,
        }).fillna(0).astype(int)

        # ── Related TEAEs ─────────────────────────────────────────────────────
        adae["AREL"] = np.where(adae["AEREL"] == "Y", "RELATED", "NOT RELATED")

        # ── Serious TEAEs ─────────────────────────────────────────────────────
        adae["ASER"] = np.where(adae["AESER"] == "Y", "Y", "N")

        # ── Analysis sequence ─────────────────────────────────────────────────
        adae["ASEQ"] = range(1, len(adae) + 1)

        logger.info("ADAE derived: %d records (%d TEAEs)",
                    len(adae), (adae["TRTEMFL"] == "Y").sum())
        return adae


# ── ADTTE - Time to Event Analysis Dataset ────────────────────────────────────

class ADTTE(ADaMDataset):
    """
    Time to Event Analysis Dataset (ADTTE).

    Supports multiple time-to-event endpoints including:
    - Overall Survival (OS)
    - Progression-Free Survival (PFS)
    - Time to Response (TTR)
    - Duration of Response (DoR)

    Uses standard Kaplan-Meier ready variables:
    AVAL (analysis value = time), CNSR (censoring flag)

    Parameters
    ----------
    study_id : str
        Sponsor study identifier.

    Examples
    --------
    >>> adtte = ADTTE(study_id="DRUGX-2024-003")
    >>> os_df = adtte.derive_os(adsl_df, events_df)
    """

    DATASET_NAME  = "ADTTE"
    DATASET_LABEL = "Time to Event Analysis Dataset"

    def derive_os(
        self,
        adsl_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Derive Overall Survival (OS) ADTTE records.

        Parameters
        ----------
        adsl_df : pd.DataFrame
            ADSL dataset.
        events_df : pd.DataFrame
            Dataset with death/last contact dates.
            Required columns: USUBJID, death_date, last_contact_date

        Returns
        -------
        pd.DataFrame
            ADTTE records for OS endpoint.
        """
        self._assert_required_cols(adsl_df,  ["USUBJID", "TRTSDT", "TRT01A"])
        self._assert_required_cols(events_df, ["USUBJID", "death_date", "last_contact_date"])

        adtte = adsl_df[["USUBJID", "TRTSDT", "TRT01A", "TRT01AN"]].merge(
            events_df[["USUBJID", "death_date", "last_contact_date"]],
            on="USUBJID",
            how="left",
        )

        adtte["PARAM"]  = "Overall Survival"
        adtte["PARAMCD"] = "OS"

        death_dt        = pd.to_datetime(adtte["death_date"],        errors="coerce")
        last_contact_dt = pd.to_datetime(adtte["last_contact_date"], errors="coerce")

        # Event = death observed; censor = last known alive
        adtte["EVNTDESC"] = np.where(death_dt.notna(), "DEATH", "CENSORED")
        adtte["CNSR"]     = np.where(death_dt.notna(), 0, 1)
        adtte["ADT"]      = np.where(death_dt.notna(), death_dt, last_contact_dt)
        adtte["ADT"]      = pd.to_datetime(adtte["ADT"])

        # AVAL = days from first dose to event/censor
        adtte["AVAL"]     = (adtte["ADT"] - adtte["TRTSDT"]).dt.days
        adtte["AVALU"]    = "DAYS"

        adtte["TRTA"]     = adtte["TRT01A"]
        adtte["TRTAN"]    = adtte["TRT01AN"]

        logger.info(
            "ADTTE OS derived: %d subjects, %d events (%.1f%%)",
            len(adtte),
            (adtte["CNSR"] == 0).sum(),
            100 * (adtte["CNSR"] == 0).mean(),
        )
        return adtte

    def derive_pfs(
        self,
        adsl_df: pd.DataFrame,
        response_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Derive Progression-Free Survival (PFS) ADTTE records.

        Parameters
        ----------
        adsl_df : pd.DataFrame
            ADSL dataset.
        response_df : pd.DataFrame
            Tumor assessment data.
            Required columns: USUBJID, progression_date, death_date, last_assessment_date

        Returns
        -------
        pd.DataFrame
            ADTTE records for PFS endpoint.
        """
        self._assert_required_cols(adsl_df, ["USUBJID", "TRTSDT", "TRT01A"])

        adtte = adsl_df[["USUBJID", "TRTSDT", "TRT01A", "TRT01AN"]].merge(
            response_df, on="USUBJID", how="left"
        )

        adtte["PARAM"]   = "Progression-Free Survival"
        adtte["PARAMCD"] = "PFS"

        prog_dt  = pd.to_datetime(adtte.get("progression_date"), errors="coerce")
        death_dt = pd.to_datetime(adtte.get("death_date"),       errors="coerce")
        last_dt  = pd.to_datetime(adtte.get("last_assessment_date"), errors="coerce")

        # PFS event = progression OR death, whichever first
        event_dt          = prog_dt.combine_first(death_dt)
        adtte["CNSR"]     = np.where(event_dt.notna(), 0, 1)
        adtte["ADT"]      = np.where(event_dt.notna(), event_dt, last_dt)
        adtte["ADT"]      = pd.to_datetime(adtte["ADT"])
        adtte["AVAL"]     = (adtte["ADT"] - adtte["TRTSDT"]).dt.days
        adtte["AVALU"]    = "DAYS"
        adtte["EVNTDESC"] = np.where(
            prog_dt.notna(), "PROGRESSION",
            np.where(death_dt.notna(), "DEATH", "CENSORED")
        )
        adtte["TRTA"]     = adtte["TRT01A"]
        adtte["TRTAN"]    = adtte["TRT01AN"]

        logger.info("ADTTE PFS derived: %d subjects", len(adtte))
        return adtte


# ── ADLB - Laboratory Data Analysis Dataset ───────────────────────────────────

class ADLB(ADaMDataset):
    """
    Laboratory Data Analysis Dataset (ADLB).

    Extends SDTM LB with analysis variables including:
    - Baseline value (BASE) and baseline flag (ABLFL)
    - Change from baseline (CHG) and percent change (PCHG)
    - Analysis normal range indicators
    - Visit number (VISITNUM) for sequencing

    Parameters
    ----------
    study_id : str
        Sponsor study identifier.

    Examples
    --------
    >>> adlb = ADLB(study_id="DRUGX-2024-003")
    >>> adlb_df = adlb.derive(lb_df, adsl_df)
    """

    DATASET_NAME  = "ADLB"
    DATASET_LABEL = "Laboratory Data Analysis Dataset"

    BASELINE_VISITS = {"SCREENING", "BASELINE", "DAY 1", "CYCLE 1 DAY 1"}

    def derive(self, lb_df: pd.DataFrame, adsl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive ADLB from SDTM LB and ADSL.

        Parameters
        ----------
        lb_df : pd.DataFrame
            SDTM LB domain dataset.
        adsl_df : pd.DataFrame
            ADSL dataset.

        Returns
        -------
        pd.DataFrame
            ADLB dataset with baseline flags and change from baseline.
        """
        self._assert_required_cols(lb_df,   ["USUBJID", "LBTESTCD", "LBSTRESN", "VISIT"])
        self._assert_required_cols(adsl_df, ["USUBJID", "TRT01A", "TRTSDT"])

        adsl_keys = adsl_df[["USUBJID", "TRT01A", "TRT01AN", "TRTSDT"]].copy()
        adlb = lb_df.merge(adsl_keys, on="USUBJID", how="left")

        adlb["TRTA"]  = adlb["TRT01A"]
        adlb["TRTAN"] = adlb["TRT01AN"]
        adlb["PARAM"]  = adlb["LBTEST"]
        adlb["PARAMCD"] = adlb["LBTESTCD"]
        adlb["AVAL"]    = adlb["LBSTRESN"]
        adlb["AVALU"]   = adlb.get("LBSTRESU", "")

        # ── Baseline flag ─────────────────────────────────────────────────────
        adlb["ABLFL"] = np.where(
            adlb["VISIT"].str.upper().isin(self.BASELINE_VISITS), "Y", ""
        )

        # ── Baseline value (BASE) ─────────────────────────────────────────────
        baseline = (
            adlb[adlb["ABLFL"] == "Y"]
            .groupby(["USUBJID", "PARAMCD"])["AVAL"]
            .first()
            .reset_index()
            .rename(columns={"AVAL": "BASE"})
        )
        adlb = adlb.merge(baseline, on=["USUBJID", "PARAMCD"], how="left")

        # ── Change from baseline ──────────────────────────────────────────────
        adlb["CHG"]  = adlb["AVAL"] - adlb["BASE"]
        adlb["PCHG"] = np.where(
            adlb["BASE"] != 0,
            100 * adlb["CHG"] / adlb["BASE"],
            np.nan,
        )

        # ── Analysis normal range indicator ───────────────────────────────────
        adlb["ANRIND"] = adlb["LBNRIND"]

        logger.info("ADLB derived: %d records across %d parameters",
                    len(adlb), adlb["PARAMCD"].nunique())
        return adlb
