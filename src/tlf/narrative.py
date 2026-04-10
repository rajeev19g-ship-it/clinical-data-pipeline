Now let's add the LLM narrative drafter:

Stay inside the tlf folder
Click "Add file" → "Create new file"
Type in the filename box:

narrative.py

Paste this code:

python"""
tlf/narrative.py
─────────────────
LLM-powered Clinical Study Report (CSR) narrative drafter.

Automatically generates ICH E3-compliant narrative text for CSR
sections from TLF outputs and ADaM summary statistics including:

    - Section 11.2  Demographics and Baseline Characteristics
    - Section 11.3  Treatments
    - Section 11.4  Analysis of Efficacy
    - Section 11.5  Safety and Tolerability

The drafter uses a structured prompt engineering approach with
domain-specific system instructions to ensure outputs follow
regulatory writing conventions (plain language, past tense,
precise statistics, no interpretation beyond the data).

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import openai
import pandas as pd

logger = logging.getLogger(__name__)


# ── Narrative output ──────────────────────────────────────────────────────────

@dataclass
class NarrativeOutput:
    """Container for a generated CSR narrative section."""

    section: str
    section_title: str
    narrative_text: str
    word_count: int
    model_used: str

    def to_dict(self) -> dict:
        return {
            "section":       self.section,
            "section_title": self.section_title,
            "narrative_text": self.narrative_text,
            "word_count":    self.word_count,
            "model_used":    self.model_used,
        }

    def save(self, output_path: str) -> None:
        """Save narrative to a plain text file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Section {self.section}: {self.section_title}\n")
            f.write("=" * 60 + "\n\n")
            f.write(self.narrative_text)
            f.write(f"\n\n[Word count: {self.word_count}]")
        logger.info("Narrative saved to %s", output_path)


# ── Narrative drafter ─────────────────────────────────────────────────────────

class CSRNarrativeDrafter:
    """
    Generates ICH E3-compliant CSR narrative sections using an LLM.

    The drafter accepts summary statistics from ADaM datasets and
    TLF outputs, formats them into structured prompts, and returns
    draft narrative text suitable for medical writer review.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env variable.
    model : str
        LLM model to use. Default gpt-4o.
    max_tokens : int
        Maximum tokens for narrative generation. Default 1500.

    Examples
    --------
    >>> drafter = CSRNarrativeDrafter()
    >>> narrative = drafter.draft_demographics(
    ...     adsl_df=adsl_df,
    ...     study_id="DRUGX-2024-003",
    ...     indication="NSCLC",
    ... )
    >>> print(narrative.narrative_text)
    """

    _SYSTEM_PROMPT = """
You are an expert regulatory medical writer with 15+ years of experience
writing Clinical Study Reports (CSRs) for FDA, EMA, and PMDA submissions.

You write narrative text for CSR sections following ICH E3 guidelines.

Your writing style:
- Past tense throughout ("were enrolled", "received", "showed")
- Plain, precise language — no marketing language or interpretation
- Statistics reported exactly as provided — do not round or modify
- Active voice where possible
- Paragraphs of 3-5 sentences
- No bullet points or headers in the narrative itself
- Do not draw conclusions beyond what the data shows
- Flag any apparent anomalies with neutral language

Always produce submission-ready draft text that a medical writer
can review and finalize with minimal editing.
""".strip()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 1500,
    ) -> None:
        self.model      = model
        self.max_tokens = max_tokens
        openai.api_key  = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def draft_demographics(
        self,
        adsl_df: pd.DataFrame,
        study_id: str,
        indication: str = "",
        phase: str = "",
    ) -> NarrativeOutput:
        """
        Draft CSR Section 11.2 — Demographics and Baseline Characteristics.

        Parameters
        ----------
        adsl_df : pd.DataFrame
            ADSL dataset (safety population).
        study_id : str
            Protocol number.
        indication : str
            Study indication (e.g. 'NSCLC').
        phase : str
            Study phase (e.g. 'Phase 3').

        Returns
        -------
        NarrativeOutput
        """
        stats = self._summarize_demographics(adsl_df)
        prompt = f"""
Study: {study_id}
Phase: {phase}
Indication: {indication}

Demographic summary statistics:
{json.dumps(stats, indent=2)}

Write the CSR Section 11.2 narrative covering:
1. Total number of subjects randomized per arm
2. Age distribution (mean, SD, range) per arm
3. Sex distribution per arm
4. Race distribution per arm
5. Any notable imbalances between arms

Length: 3-4 paragraphs.
""".strip()

        return self._call_llm(
            prompt=prompt,
            section="11.2",
            section_title="Demographic and Baseline Characteristics",
        )

    def draft_safety_summary(
        self,
        adae_df: pd.DataFrame,
        adsl_df: pd.DataFrame,
        study_id: str,
    ) -> NarrativeOutput:
        """
        Draft CSR Section 11.5 — Safety and Tolerability summary.

        Parameters
        ----------
        adae_df : pd.DataFrame
            ADAE dataset (treatment-emergent AEs).
        adsl_df : pd.DataFrame
            ADSL dataset (for denominators).
        study_id : str
            Protocol number.

        Returns
        -------
        NarrativeOutput
        """
        stats = self._summarize_safety(adae_df, adsl_df)
        prompt = f"""
Study: {study_id}

Safety summary statistics (TEAEs):
{json.dumps(stats, indent=2)}

Write the CSR Section 11.5 safety narrative covering:
1. Overall TEAE incidence per arm
2. Serious TEAEs per arm
3. Related TEAEs per arm
4. Grade 3+ TEAEs per arm
5. Most frequently reported TEAEs (top 5 by preferred term)
6. Deaths and discontinuations due to AEs if any

Length: 4-5 paragraphs.
""".strip()

        return self._call_llm(
            prompt=prompt,
            section="11.5",
            section_title="Safety and Tolerability",
        )

    def draft_efficacy_summary(
        self,
        adtte_df: pd.DataFrame,
        study_id: str,
        primary_endpoint: str = "Overall Survival",
    ) -> NarrativeOutput:
        """
        Draft CSR Section 11.4 — Efficacy summary.

        Parameters
        ----------
        adtte_df : pd.DataFrame
            ADTTE dataset for the primary endpoint.
        study_id : str
            Protocol number.
        primary_endpoint : str
            Name of the primary endpoint.

        Returns
        -------
        NarrativeOutput
        """
        stats = self._summarize_tte(adtte_df, primary_endpoint)
        prompt = f"""
Study: {study_id}
Primary endpoint: {primary_endpoint}

Time-to-event summary statistics:
{json.dumps(stats, indent=2)}

Write the CSR Section 11.4 efficacy narrative covering:
1. Number of subjects in the ITT/efficacy population per arm
2. Number of events and censored subjects per arm
3. Median time to event with 95% CI per arm (if available)
4. Overall event rate per arm
5. Brief statement on the direction of the treatment effect
   (descriptive only — do not use words like "significant" or "superior")

Length: 3-4 paragraphs.
""".strip()

        return self._call_llm(
            prompt=prompt,
            section="11.4",
            section_title="Analysis of Efficacy",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_llm(
        self,
        prompt: str,
        section: str,
        section_title: str,
    ) -> NarrativeOutput:
        """Send prompt to LLM and return NarrativeOutput."""
        try:
            response = openai.chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            logger.info(
                "Narrative drafted: Section %s (%d words)",
                section, len(text.split()),
            )
            return NarrativeOutput(
                section=section,
                section_title=section_title,
                narrative_text=text,
                word_count=len(text.split()),
                model_used=self.model,
            )
        except openai.OpenAIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise

    @staticmethod
    def _summarize_demographics(adsl_df: pd.DataFrame) -> dict:
        """Compute demographic summary statistics per treatment arm."""
        trt = "TRTA" if "TRTA" in adsl_df.columns else "TRT01A"
        summary = {}
        for arm, grp in adsl_df.groupby(trt):
            summary[arm] = {
                "n": len(grp),
                "age_mean":   round(grp["AGE"].mean(), 1) if "AGE" in grp else None,
                "age_sd":     round(grp["AGE"].std(),  2) if "AGE" in grp else None,
                "age_median": round(grp["AGE"].median(), 1) if "AGE" in grp else None,
                "age_min":    int(grp["AGE"].min()) if "AGE" in grp else None,
                "age_max":    int(grp["AGE"].max()) if "AGE" in grp else None,
                "sex_m_n":    int((grp.get("SEX") == "M").sum()),
                "sex_f_n":    int((grp.get("SEX") == "F").sum()),
                "sex_m_pct":  round(100 * (grp.get("SEX") == "M").mean(), 1),
                "sex_f_pct":  round(100 * (grp.get("SEX") == "F").mean(), 1),
            }
        return summary

    @staticmethod
    def _summarize_safety(adae_df: pd.DataFrame, adsl_df: pd.DataFrame) -> dict:
        """Compute TEAE summary statistics per treatment arm."""
        trt = "TRTA" if "TRTA" in adae_df.columns else "TRT01A"
        teae = adae_df[adae_df.get("TRTEMFL", pd.Series("Y")) == "Y"]
        n_arm = adsl_df.groupby(
            "TRTA" if "TRTA" in adsl_df.columns else "TRT01A"
        ).size().to_dict()

        summary = {}
        for arm, n in n_arm.items():
            arm_teae = teae[teae[trt] == arm]
            n_subj   = arm_teae["USUBJID"].nunique()

            top5 = (
                arm_teae.groupby("AEDECOD")["USUBJID"]
                .nunique()
                .nlargest(5)
                .to_dict()
            )

            summary[arm] = {
                "n_safety":          n,
                "n_any_teae":        n_subj,
                "pct_any_teae":      round(100 * n_subj / n, 1) if n > 0 else 0,
                "n_serious":         int(arm_teae[arm_teae.get("ASER", pd.Series("N")) == "Y"]["USUBJID"].nunique()),
                "n_related":         int(arm_teae[arm_teae.get("AEREL", pd.Series("N")) == "Y"]["USUBJID"].nunique()),
                "n_grade3plus":      int(arm_teae[arm_teae.get("ATOXGR", pd.Series(0)) >= 3]["USUBJID"].nunique()),
                "top5_preferred_terms": top5,
            }
        return summary

    @staticmethod
    def _summarize_tte(adtte_df: pd.DataFrame, endpoint: str) -> dict:
        """Compute time-to-event summary statistics per treatment arm."""
        trt = "TRTA" if "TRTA" in adtte_df.columns else "TRT01A"
        param_df = adtte_df[adtte_df.get("PARAMCD", adtte_df.get("PARAM", "")) == endpoint] \
            if endpoint in adtte_df.get("PARAMCD", pd.Series()).values \
            else adtte_df

        summary = {}
        for arm, grp in param_df.groupby(trt):
            n_events   = int((grp["CNSR"] == 0).sum())
            n_censored = int((grp["CNSR"] == 1).sum())
            summary[arm] = {
                "n":           len(grp),
                "n_events":    n_events,
                "n_censored":  n_censored,
                "event_rate":  round(100 * n_events / len(grp), 1) if len(grp) > 0 else 0,
                "median_aval": round(grp["AVAL"].median(), 1) if "AVAL" in grp else None,
                "min_aval":    round(grp["AVAL"].min(), 1)    if "AVAL" in grp else None,
                "max_aval":    round(grp["AVAL"].max(), 1)    if "AVAL" in grp else None,
            }
        return summary
