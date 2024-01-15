"""Some utilities/wrappers for Qiskit"""


def to_sec(value: float | int, unit: str) -> float:
    multiplier = None
    if unit == "ns":
        multiplier = 1e-9
    elif unit == "us":
        multiplier = 1e-6
    elif unit == "ms":
        multiplier = 1e-3
    elif unit == "s":
        multiplier = 1.0

    if multiplier is None:
        raise RuntimeError(f"unknown unit: {unit}")

    return float(value) * multiplier
