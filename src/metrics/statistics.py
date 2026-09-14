import pandas as pd


def summary_metrics(detections: pd.DataFrame) -> dict:
    scores = (
        detections["score"]
        if "score" in detections.columns
        else pd.Series(dtype=float)
    )

    return {
        "tree_count": int(len(detections)),
        "mean_confidence": float(scores.mean()) if not scores.empty else None,
    }
