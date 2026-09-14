from datetime import datetime, timezone


def build_manifest(**values) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
