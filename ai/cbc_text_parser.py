
import re

from ai.cbc_analyzer import analyze_cbc


# Explicit aliases only; add more after testing real report formats.
CBC_ALIASES = {
    "hemoglobin": "Hemoglobin",
    "hb": "Hemoglobin",
    "wbc": "WBC",
    "white blood cell count": "WBC",
    "platelets": "Platelets",
    "platelet count": "Platelets",
    "rbc": "RBC",
    "red blood cell count": "RBC",
    "hematocrit": "Hematocrit",
    "hct": "Hematocrit",
}

ROW_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Za-z ]+?)\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s+"
    r"(?P<unit>\S+)\s+"
    r"(?P<low>-?\d+(?:\.\d+)?)\s*[-–]\s*"
    r"(?P<high>-?\d+(?:\.\d+)?)\s*$"
)


def parse_cbc_text(text: str) -> dict:
    """Parse supported CBC rows with explicit numeric reference ranges."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("CBC report text is empty.")

    measurements = {}

    for line in text.splitlines():
        match = ROW_PATTERN.match(line)
        if not match:
            continue

        name = " ".join(match.group("name").lower().split())
        canonical_name = CBC_ALIASES.get(name)

        if canonical_name is None:
            continue

        if canonical_name in measurements:
            raise ValueError(
                f"Duplicate measurement found: {canonical_name}"
            )

        measurements[canonical_name] = {
            "value": float(match.group("value")),
            "unit": match.group("unit"),
            "low": float(match.group("low")),
            "high": float(match.group("high")),
        }

    if not measurements:
        raise ValueError(
            "No supported CBC measurements with explicit reference "
            "ranges were found. Review the report manually."
        )

    # Reuse the existing range-comparison logic.
    return analyze_cbc(measurements)
