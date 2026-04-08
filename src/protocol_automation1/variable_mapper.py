"""
protocol_automation/variable_mapper.py
───────────────────────────────────────
Maps protocol-extracted variables to CDISC SDTM domains and ADaM datasets.

Given a parsed ProtocolMetadata object, this module:
- Suggests the most likely SDTM domain for each endpoint/assessment
- Generates a draft variable mapping table (CSV-ready)
- Flags any endpoints that may require custom domains (SUPPQUAL)

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

from .parser import ProtocolMetadata

logger = logging.getLogger(__name__)


# ── SDTM domain reference ─────────────────────────────────────────────────────

SDTM_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "AE": ["adverse event", "adverse experience", "toxicity", "safety"],
    "CM": ["concomitant", "medication", "drug", "treatment", "therapy"],
    "DM": ["demographic", "age", "sex", "gender", "race", "ethnicity", "subject"],
    "DS": ["disposition", "withdrawal", "discontinuation", "completion"],
    "EX": ["exposure", "dose", "administration", "infusion", "injection"],
    "LB": ["laboratory", "lab", "hematology", "chemistry", "urinalysis", "biomarker"],
    "MH": ["medical history", "prior condition", "comorbidity", "history"],
    "PE": ["physical examination", "physical exam", "vital signs"],
    "QS": ["questionnaire", "scale", "score", "patient reported", "PRO"],
    "RS": ["response", "tumor", "recist", "overall response", "remission"],
    "TU": ["tumor", "lesion", "imaging", "scan", "mri", "ct scan"],
    "VS": ["vital sign", "blood pressure", "heart rate", "temperature", "weight", "bmi"],
}

ADAM_DATASET_KEYWORDS: dict[str, list[str]] = {
    "ADSL":  ["subject", "demographic", "baseline", "disposition", "population flag"],
    "ADAE":  ["adverse event", "toxicity", "safety"],
    "ADLB":  ["laboratory", "lab", "biomarker", "hematology", "chemistry"],
    "ADTTE": ["time to event", "survival", "progression", "event free", "overall survival"],
    "ADRS":  ["response", "tumor response", "recist", "overall response"],
    "ADCM":  ["concomitant", "medication", "prior therapy"],
    "ADVS":  ["vital sign", "blood pressure", "heart rate", "weight"],
    "ADQS":  ["questionnaire", "scale", "score", "PRO"],
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VariableMapping:
    """A single endpoint-to-domain mapping record."""

    endpoint: str
    endpoint_type: str          # primary | secondary | inclusion | exclusion
    suggested_sdtm_domain: str
    suggested_adam_dataset: str
    confidence: str             # high | medium | low
    requires_suppqual: bool = False
    notes: str = ""


@dataclass
class MappingTable:
    """Full mapping table for a protocol."""

    protocol_number: str
    study_title: str
    mappings: list[VariableMapping] = field(default_factory=list)

    def to_csv(self) -> str:
        """Export mapping table as a CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "endpoint",
            "endpoint_type",
            "suggested_sdtm_domain",
            "suggested_adam_dataset",
            "confidence",
            "requires_suppqual",
            "notes",
        ])
        for m in self.mappings:
            writer.writerow([
                m.endpoint,
                m.endpoint_type,
                m.suggested_sdtm_domain,
                m.suggested_adam_dataset,
                m.confidence,
                m.requires_suppqual,
                m.notes,
            ])
        return output.getvalue()

    def summary(self) -> dict:
        """Return a high-level summary of the mapping table."""
        domains = [m.suggested_sdtm_domain for m in self.mappings]
        return {
            "total_endpoints": len(self.mappings),
            "unique_sdtm_domains": sorted(set(domains)),
            "suppqual_required": sum(1 for m in self.mappings if m.requires_suppqual),
            "low_confidence_count": sum(1 for m in self.mappings if m.confidence == "low"),
        }


# ── Mapper ────────────────────────────────────────────────────────────────────

class VariableMapper:
    """
    Maps protocol endpoints and criteria to CDISC SDTM domains
    and ADaM datasets using keyword matching with confidence scoring.

    Parameters
    ----------
    custom_rules : dict, optional
        Additional keyword-to-domain rules to extend the built-in reference.
        Format: {"DOMAIN": ["keyword1", "keyword2"]}

    Examples
    --------
    >>> from src.protocol_automation.parser import ProtocolParser
    >>> from src.protocol_automation.variable_mapper import VariableMapper
    >>>
    >>> metadata = ProtocolParser().parse("protocol.pdf")
    >>> mapper = VariableMapper()
    >>> table = mapper.map(metadata)
    >>> print(table.to_csv())
    """

    def __init__(self, custom_rules: Optional[dict[str, list[str]]] = None) -> None:
        self.sdtm_rules = {**SDTM_DOMAIN_KEYWORDS, **(custom_rules or {})}

    # ── Public API ────────────────────────────────────────────────────────────

    def map(self, metadata: ProtocolMetadata) -> MappingTable:
        """
        Generate a full variable mapping table from parsed protocol metadata.

        Parameters
        ----------
        metadata : ProtocolMetadata
            Output from ProtocolParser.parse().

        Returns
        -------
        MappingTable
            Dataclass containing all VariableMapping records.
        """
        table = MappingTable(
            protocol_number=metadata.protocol_number,
            study_title=metadata.study_title,
        )

        endpoint_groups = [
            (metadata.primary_endpoints,   "primary"),
            (metadata.secondary_endpoints, "secondary"),
            (metadata.inclusion_criteria,  "inclusion"),
            (metadata.exclusion_criteria,  "exclusion"),
        ]

        for endpoints, ep_type in endpoint_groups:
            for endpoint in endpoints:
                mapping = self._map_single(endpoint, ep_type)
                table.mappings.append(mapping)
                logger.debug("Mapped [%s] → %s (%s)", ep_type, mapping.suggested_sdtm_domain, mapping.confidence)

        logger.info(
            "Mapping complete: %d records across %d SDTM domains",
            len(table.mappings),
            len(table.summary()["unique_sdtm_domains"]),
        )
        return table

    # ── Private helpers ───────────────────────────────────────────────────────

    def _map_single(self, endpoint: str, endpoint_type: str) -> VariableMapping:
        """Map a single endpoint string to its most likely SDTM domain and ADaM dataset."""
        text = endpoint.lower()

        sdtm_domain, sdtm_confidence = self._match_domain(text, self.sdtm_rules)
        adam_dataset, _ = self._match_domain(text, ADAM_DATASET_KEYWORDS)

        requires_suppqual = sdtm_confidence == "low"
        notes = "Consider SUPPQUAL for non-standard variable" if requires_suppqual else ""

        return VariableMapping(
            endpoint=endpoint,
            endpoint_type=endpoint_type,
            suggested_sdtm_domain=sdtm_domain,
            suggested_adam_dataset=adam_dataset,
            confidence=sdtm_confidence,
            requires_suppqual=requires_suppqual,
            notes=notes,
        )

    @staticmethod
    def _match_domain(text: str, rules: dict[str, list[str]]) -> tuple[str, str]:
        """
        Score a text string against domain keyword rules.

        Returns
        -------
        tuple[str, str]
            (best_domain, confidence) where confidence is high | medium | low.
        """
        scores: dict[str, int] = {}

        for domain, keywords in rules.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[domain] = score

        if not scores:
            return "UNKNOWN", "low"

        best_domain = max(scores, key=lambda d: scores[d])
        best_score = scores[best_domain]

        if best_score >= 2:
            confidence = "high"
        elif best_score == 1:
            confidence = "medium"
        else:
            confidence = "low"

        return best_domain, confidence
