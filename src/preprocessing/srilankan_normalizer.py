"""
Sri Lankan Normalizer

Loads Sri Lankan slang/domain mappings from JSON.
Supports multi-word phrase replacements and regex word-boundary normalization.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

import json
from pathlib import Path
import re
from typing import Optional, List

from src.utils.logger import logger


class SriLankanNormalizer:

    def __init__(self):

        # Project Root
        project_root = Path(__file__).resolve().parents[2]

        dictionary_path = (
            project_root
            / "resources"
            / "slang_dictionary.json"
        )

        if not dictionary_path.exists():
            raise FileNotFoundError(
                f"Slang dictionary not found:\n{dictionary_path}"
            )

        with open(dictionary_path, "r", encoding="utf-8") as f:
            slang_groups = json.load(f)

        self.multi_word_map = {}
        self.single_word_map = {}

        for standard_word, variations in slang_groups.items():

            std_clean = standard_word.strip()

            # Map standard word
            if " " in std_clean:
                self.multi_word_map[std_clean.lower()] = std_clean
            else:
                self.single_word_map[std_clean.lower()] = std_clean

            # Map all variations
            for variation in variations:
                var_clean = variation.strip()
                if " " in var_clean:
                    self.multi_word_map[var_clean.lower()] = std_clean
                else:
                    self.single_word_map[var_clean.lower()] = std_clean

        # Sort multi-word keys by descending length for greedy matching
        sorted_multi_words = sorted(
            self.multi_word_map.keys(),
            key=len,
            reverse=True
        )

        if sorted_multi_words:
            escaped_phrases = [
                re.escape(phrase) for phrase in sorted_multi_words
            ]
            self.multi_word_pattern = re.compile(
                r"(?<![a-zA-Z0-9\u0D80-\u0DFF])("
                + "|".join(escaped_phrases)
                + r")(?![a-zA-Z0-9\u0D80-\u0DFF])",
                re.IGNORECASE
            )
        else:
            self.multi_word_pattern = None

        # Regex for single words (including Sinhala Unicode range)
        self.single_word_pattern = re.compile(
            r"[a-zA-Z0-9\u0D80-\u0DFF]+"
        )

        total_mappings = len(self.single_word_map) + len(self.multi_word_map)
        logger.info(
            f"Loaded {total_mappings} Sri Lankan mappings."
        )

    def normalize(self, text: Optional[str]) -> str:

        if not text:
            return ""

        # 1. Multi-word phrase replacements
        if self.multi_word_pattern:
            text = self.multi_word_pattern.sub(
                lambda m: self.multi_word_map.get(m.group(0).lower(), m.group(0)),
                text
            )

        # 2. Single-word token replacements
        text = self.single_word_pattern.sub(
            lambda m: self.single_word_map.get(m.group(0).lower(), m.group(0)),
            text
        )

        return text

    def normalize_batch(self, texts: List[str]) -> List[str]:

        return [
            self.normalize(text)
            for text in texts
        ]
