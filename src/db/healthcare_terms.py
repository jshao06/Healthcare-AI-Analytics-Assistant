"""
healthcare_terms.py

At this stage, it (variable - concepts) only defines a common interface 
and includes a small built-in healthcare concepts.

In the future, it will be replaced or extended.
It will include the entire ICD-10, LOINC, or RxNorm datasets,
containing tens or hundreds of thousands of concepts.

Author: Jianhua Shao
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import re

@dataclass(frozen=True)
class MedicalConcept:
    """
    Represents one clinical concept.
    """    
    concept: str
    code: str
    vocabulary: str
    synonyms: tuple[str, ...]

class HealthcareTerms:

    def __init__(self) -> None:
        self._concepts: dict[str, MedicalConcept] = {}
        self._lookup: dict[str, list[MedicalConcept]] = defaultdict(list)

        self._load_default_terms()

    def _load_default_terms(self) -> None:

        concepts = [

            MedicalConcept(
                concept="hypertension",
                code="I10",
                vocabulary="ICD10",
                synonyms=(
                    "high blood pressure",
                    "bp",
                    "htn",
                ),
            ),

            MedicalConcept(
                concept="diabetes mellitus",
                code="E11",
                vocabulary="ICD10",
                synonyms=(
                    "diabetes",
                    "dm",
                    "type 2 diabetes",
                    "t2dm",
                ),
            ),

            MedicalConcept(
                concept="myocardial infarction",
                code="I21",
                vocabulary="ICD10",
                synonyms=(
                    "heart attack",
                    "mi",
                ),
            ),

            MedicalConcept(
                concept="stroke",
                code="I63",
                vocabulary="ICD10",
                synonyms=(
                    "cva",
                    "cerebrovascular accident",
                ),
            ),

            MedicalConcept(
                concept="creatinine",
                code="2160-0",
                vocabulary="LOINC",
                synonyms=(
                    "serum creatinine",
                ),
            ),

            MedicalConcept(
                concept="glucose",
                code="2345-7",
                vocabulary="LOINC",
                synonyms=(
                    "blood sugar",
                    "blood glucose",
                ),
            ),
        ]

        for concept in concepts:
            self.add(concept)

    def add(self, concept: MedicalConcept) -> None:

        self._concepts[concept.concept] = concept

        self._lookup[concept.concept.lower()].append(concept)

        for synonym in concept.synonyms:
            self._lookup[synonym.lower()].append(concept)

    def find(self, text: str) -> list[MedicalConcept]:
        return self._lookup.get(text.lower(), [])

    def find_in_text(self, text: str) -> list[MedicalConcept]:
        """Return concepts mentioned by a term or phrase within *text*."""
        normalized_text = text.lower().strip()
        matches: dict[tuple[str, str], MedicalConcept] = {}

        for term, concepts in self._lookup.items():
            pattern = rf'(?<!\w){re.escape(term)}(?!\w)'
            if re.search(pattern, normalized_text) is None:
                continue

            for concept in concepts:
                matches[(concept.vocabulary, concept.code)] = concept

        return list(matches.values())

    def expand(self, text: str) -> set[str]:
        """Expand clinical phrases into concepts, synonyms, and codes."""
        expanded = set(text.lower().split())

        for concept in self.find_in_text(text):
            expanded.add(concept.concept.lower())
            expanded.add(concept.code.lower())
            expanded.update(synonym.lower() for synonym in concept.synonyms)

        return expanded
