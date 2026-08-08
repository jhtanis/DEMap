"""Text normalization utilities.

These helpers are intentionally *conservative* and are currently used for
dataset **de-duplication keys**, not for rewriting stored query text.

Design goals
------------
- Preserve case (**no lowercasing**), because casing can be meaningful
  (e.g., acronyms) and provenance/family decisions may depend on it.
- Collapse obvious formatting noise:
  - leading/trailing whitespace
  - multiple whitespace characters
  - common Unicode “smart quotes” and dash/minus variants

The normalizer is designed to be:
- deterministic
- idempotent
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_WS_RE = re.compile(r"\s+")


# Map common Unicode quote variants to their ASCII equivalents.
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201A": "'",  # single low-9 quotation mark
        "\u201B": "'",  # single high-reversed-9 quotation mark
        "\u2032": "'",  # prime
        "\u2035": "'",  # reversed prime
        "\u201C": '"',  # left double quotation mark
        "\u201D": '"',  # right double quotation mark
        "\u201E": '"',  # double low-9 quotation mark
        "\u201F": '"',  # double high-reversed-9 quotation mark
        "\u00AB": '"',  # left-pointing double angle quotation mark
        "\u00BB": '"',  # right-pointing double angle quotation mark
    }
)


# Map common dash/minus variants to a plain hyphen.
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        "\u2043": "-",  # hyphen bullet
    }
)


def normalize_query_text(x: Any) -> str:
    """Normalize query text for de-duplication.

    Parameters
    ----------
    x
        Any object; None becomes an empty string.

    Returns
    -------
    str
        Normalized text. **Case is preserved**.
    """
    if x is None:
        return ""

    s = str(x)

    # Replace NBSP with space.
    s = s.replace("\u00A0", " ")

    # Normalize compatibility characters (does not lowercase).
    s = unicodedata.normalize("NFKC", s)

    # Normalize punctuation.
    s = s.translate(_QUOTE_TRANSLATION)
    s = s.translate(_DASH_TRANSLATION)

    # Trim + collapse whitespace.
    s = s.strip()
    s = _WS_RE.sub(" ", s)

    return s
