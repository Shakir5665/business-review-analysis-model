"""
Text Cleaner

Performs basic text cleaning before NLP preprocessing.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

import html
import unicodedata
from typing import Optional

from configs.preprocessing_config import (
    LOWERCASE,
    REMOVE_URLS,
    REMOVE_EMAILS,
    REMOVE_HTML,
    REMOVE_MENTIONS,
    REMOVE_EXTRA_WHITESPACE,
    NORMALIZE_UNICODE,
    REMOVE_CONTROL_CHARACTERS,
    NORMALIZE_HASHTAGS,
    URL_PATTERN,
    EMAIL_PATTERN,
    HTML_PATTERN,
    MENTION_PATTERN,
    HASHTAG_PATTERN,
    WHITESPACE_PATTERN,
    CONTROL_CHARACTER_PATTERN,
    MULTIPLE_DOTS_PATTERN,
    MULTIPLE_EXCLAMATION_PATTERN,
    MULTIPLE_QUESTION_PATTERN,
    UNICODE_WHITESPACE_PATTERN,
    # Dataset-specific patterns
    BACKSLASH_PATTERN,
    SQUARE_BRACKET_PATTERN,
    QUOTE_PATTERN,
   
     # Dataset-specific options
    REMOVE_BACKSLASHES,
    REMOVE_SQUARE_BRACKETS,
    REMOVE_QUOTES,
    SPACE_PUNCTUATION_AND_SYMBOLS,
    PUNCTUATION_AND_SYMBOLS_PATTERN,
)

from src.utils.logger import logger


class TextCleaner:
    """
    Performs low-level text cleaning.

    Responsibilities:
        - Unicode normalization
        - Lowercasing
        - HTML removal
        - URL removal
        - Email removal
        - Mention removal
        - Hashtag normalization
        - Whitespace normalization

    Does NOT perform:
        - Emoji conversion
        - Sinhala conversion
        - Sri Lankan normalization
    """

    def __init__(self):
        logger.info("TextCleaner initialized")

    def clean(self, text: Optional[str]) -> str:

        if text is None:
            return ""

        text = str(text)

        # -------------------------------------------------
        # Unicode normalization
        # -------------------------------------------------

        if NORMALIZE_UNICODE:
            text = unicodedata.normalize("NFC", text)

        # -------------------------------------------------
        # Decode HTML entities
        # -------------------------------------------------

        text = html.unescape(text)

        # -------------------------------------------------
        # Lowercase
        # -------------------------------------------------

        if LOWERCASE:
            text = text.lower()

        # -------------------------------------------------
        # Remove HTML tags
        # -------------------------------------------------

        if REMOVE_HTML:
            text = HTML_PATTERN.sub(" ", text)

        # -------------------------------------------------
        # Remove URLs
        # -------------------------------------------------

        if REMOVE_URLS:
            text = URL_PATTERN.sub(" ", text)

        # -------------------------------------------------
        # Remove Emails
        # -------------------------------------------------

        if REMOVE_EMAILS:
            text = EMAIL_PATTERN.sub(" ", text)

        # -------------------------------------------------
        # Remove Mentions
        # -------------------------------------------------

        if REMOVE_MENTIONS:
            text = MENTION_PATTERN.sub(" ", text)

        # -------------------------------------------------
        # Normalize hashtags
        # Example:
        # #delivery -> delivery
        # -------------------------------------------------

        if NORMALIZE_HASHTAGS:
            text = HASHTAG_PATTERN.sub(r"\1", text)

        # -------------------------------------------------
        # Remove Unicode whitespace
        # -------------------------------------------------

        text = UNICODE_WHITESPACE_PATTERN.sub(" ", text)

        # -------------------------------------------------
        # Remove control characters
        # -------------------------------------------------

        if REMOVE_CONTROL_CHARACTERS:
            text = CONTROL_CHARACTER_PATTERN.sub(" ", text)

        # -------------------------------------------------
        # Space punctuation, symbols, and emojis
        # -------------------------------------------------

        if SPACE_PUNCTUATION_AND_SYMBOLS:
            text = PUNCTUATION_AND_SYMBOLS_PATTERN.sub(r" \1 ", text)

        # -------------------------------------------------
        # Normalize punctuation
        # -------------------------------------------------

        text = MULTIPLE_DOTS_PATTERN.sub(".", text)

        text = MULTIPLE_EXCLAMATION_PATTERN.sub("!", text)

        text = MULTIPLE_QUESTION_PATTERN.sub("?", text)

        # -------------------------------------------------
        # Normalize whitespace
        # -------------------------------------------------

        if REMOVE_EXTRA_WHITESPACE:
            text = WHITESPACE_PATTERN.sub(" ", text)
        
        # Remove backslashes
        if REMOVE_BACKSLASHES:
            text = BACKSLASH_PATTERN.sub(" ", text)

        # Remove square brackets
        if REMOVE_SQUARE_BRACKETS:
            text = SQUARE_BRACKET_PATTERN.sub(" ", text)

        # Remove quotes
        if REMOVE_QUOTES:
            text = QUOTE_PATTERN.sub(" ", text)

        return text.strip()

    def clean_batch(self, texts):

        return [self.clean(text) for text in texts]
    
     