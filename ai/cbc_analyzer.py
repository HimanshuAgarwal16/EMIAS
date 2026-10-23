def analyze_cbc(results):
    """
    Compare CBC measurements with laboratory-provided reference ranges.

    Expected format:
    {
        "Hemoglobin": {
            "value": 13.5,
            "unit": "g/dL",
            "low": 12.0,
            "high": 16.0
        }
    }
    """
    if not isinstance(results, dict) or not results:
        raise ValueError("Provide at least one CBC measurement.")

    findings = []

    for name, item in results.items():
        if not isinstance(item, dict):
            raise ValueError(f"Invalid measurement format: {name}")

        value = item.get("value")
        low = item.get("low")
        high = item.get("high")
        unit = item.get("unit", "")

        if any(
            isinstance(number, bool) or not isinstance(number, (int, float))
            for number in (value, low, high)
        ):
            raise ValueError(
                f"{name}: value, low, and high must be numeric."
            )

        if low > high:
            raise ValueError(f"{name}: lower limit exceeds upper limit.")

        if value < low:
            status = "Below supplied reference range"
        elif value > high:
            status = "Above supplied reference range"
        else:
            status = "Within supplied reference range"

        findings.append({
            "test": name,
            "value": value,
            "unit": unit,
            "reference_range": f"{low}–{high}",
            "status": status,
        })

    return {
        "analysis_type": "CBC reference-range comparison",
        "findings": findings,
        "disclaimer": (
            "Not a diagnosis. Interpret results using the original "
            "laboratory report and qualified clinical judgment."
        ),
    }
