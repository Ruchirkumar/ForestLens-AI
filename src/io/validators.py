from pathlib import Path


def validate_extension(path: str | Path, allowed: set[str]) -> None:
    suffix = Path(path).suffix.lower()

    if suffix not in allowed:
        raise ValueError(
            f"Unsupported file type: {suffix or '<none>'}. "
            f"Allowed: {sorted(allowed)}"
        )
