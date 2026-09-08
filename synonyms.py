"""
synonyms.py

Contains medical vocabulary.

Author: Jianhua Shao
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable

class SynonymManager:
    """
    Manages bidirectional synonyms for schema search.

    Example:
        "bp" <-> "blood pressure"
        "htn" <-> "hypertension"
        "heart attack" <-> "myocardial infarction"
    """

    def __init__(self) -> None:
        self._synonyms: dict[str, set[str]] = defaultdict(set)
        self._load_default_synonyms()

    def _load_default_synonyms(self) -> None:
        """
        Load built-in healthcare synonyms.
        """

        synonym_groups = [

            # Hypertension
            [
                "hypertension",
                "high blood pressure",
                "blood pressure",
                "bp",
                "htn",
            ],

            # Diabetes
            [
                "diabetes",
                "diabetes mellitus",
                "dm",
                "t2dm",
                "type 2 diabetes",
            ],

            # Heart Attack
            [
                "heart attack",
                "myocardial infarction",
                "mi",
            ],

            # Stroke
            [
                "stroke",
                "cerebrovascular accident",
                "cva",
            ],

            # Kidney
            [
                "kidney",
                "renal",
            ],

            # Glucose
            [
                "glucose",
                "blood sugar",
                "blood glucose",
            ],

            # Creatinine
            [
                "creatinine",
                "serum creatinine",
            ],

            # Male/Female
            [
                "male",
                "man",
                "m",
            ],

            [
                "female",
                "woman",
                "f",
            ],

            # Emergency
            [
                "emergency",
                "er",
                "ed",
                "emergency department",
            ],

            # Medication
            [
                "drug",
                "medicine",
                "medication",
                "prescription",
            ],
        ]

        for group in synonym_groups:
            self.add_group(group)

    def add_group(self, words: Iterable[str]) -> None:
        """
        Adds a group of equivalent words.
        """
        
        normalized = {
            word.strip().lower()
            for word in words
            if word.strip()
        }  

        for word in normalized:
            self._synonyms[word].update(normalized - {word})

    def get_synonyms(self, term: str) -> list[str]:
        """
        Returns all synonyms for a term.

        Example:
            get_synonyms("bp")

        Returns:
            [
                "blood pressure",
                "hypertension",
                "htn",
                "high blood pressure"
            ]
        """
        term = term.lower().strip()

        return sorted(self._synonyms.get(term, set()))

    def expand(self, text: str) -> set[str]:
        """
        Expand a search query into related terms.

        Example:
            "bp patient"

        Returns:
            {
                "bp",
                "blood pressure",
                "hypertension",
                "high blood pressure",
                "patient"
            }
        """

        expanded: set[str] = set()

        normalized_text = text.lower().strip()

        for token in normalized_text.split():
            expanded.add(token)

        for term, synonyms in self._synonyms.items():
            pattern = rf'(?<!\w){re.escape(term)}(?!\w)'
            if re.search(pattern, normalized_text) is not None:
                expanded.add(term)
                expanded.update(synonyms)

        return expanded

    def has_term(self, term: str) -> bool:
        return term.lower() in self._synonyms

    def __len__(self) -> int:
        return len(self._synonyms)
