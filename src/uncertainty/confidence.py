def confidence_bands(scores):
    return {
        "high": sum(1 for score in scores if score >= 0.70),
        "medium": sum(1 for score in scores if 0.40 <= score < 0.70),
        "low": sum(1 for score in scores if score < 0.40),
    }
