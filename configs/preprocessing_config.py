"""
Preprocessing Configuration

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

import re

# ==========================================================
# Cleaning Options
# ==========================================================

LOWERCASE = True

REMOVE_URLS = True

REMOVE_EMAILS = True

REMOVE_HTML = True

REMOVE_MENTIONS = True

REMOVE_EXTRA_WHITESPACE = True

NORMALIZE_UNICODE = True

REMOVE_CONTROL_CHARACTERS = True

NORMALIZE_HASHTAGS = True

KEEP_NUMBERS = True

KEEP_UNDERSCORE = True

MAX_REPEATED_CHARACTERS = 2

REMOVE_BACKSLASHES = True

REMOVE_BRACKETS = True

REMOVE_SQUARE_BRACKETS = True

REMOVE_QUOTES = True

SPACE_PUNCTUATION_AND_SYMBOLS = True


# ==========================================================
# Regex Patterns
# ==========================================================

PUNCTUATION_AND_SYMBOLS_PATTERN = re.compile(
    r"([^\s\w\u0D80-\u0DFF])"
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

HTML_PATTERN = re.compile(
    r"<.*?>"
)

MENTION_PATTERN = re.compile(
    r"@\w+"
)

HASHTAG_PATTERN = re.compile(
    r"#(\w+)"
)

WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

CONTROL_CHARACTER_PATTERN = re.compile(
    r"[\r\n\t]"
)

MULTIPLE_DOTS_PATTERN = re.compile(
    r"\.{2,}"
)

MULTIPLE_EXCLAMATION_PATTERN = re.compile(
    r"!{2,}"
)

MULTIPLE_QUESTION_PATTERN = re.compile(
    r"\?{2,}"
)

REPEATED_CHARACTER_PATTERN = re.compile(
    r"(.)\1{2,}"
)

UNICODE_WHITESPACE_PATTERN = re.compile(
    r"[\u2000-\u200B\u200E-\u200F\u2028-\u202F]"
)


# ==========================================================
# Dataset-Specific Patterns
# ==========================================================

BACKSLASH_PATTERN = re.compile(r"\\")

SQUARE_BRACKET_PATTERN = re.compile(r"[\[\]]")

QUOTE_PATTERN = re.compile(r"[\'\"]")

NESTED_STRUCTURE_PATTERN = re.compile(r"[\'\"\[\]\\]")

EXTRA_SPACE_PATTERN = re.compile(r"\s+")