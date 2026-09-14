import pandas as pd

from src.uncertainty.sensitivity import threshold_sensitivity


def test_threshold_sensitivity():
    predictions = pd.DataFrame(
        {"score": [0.25, 0.35, 0.45, 0.55, 0.75]}
    )

    rows = threshold_sensitivity(predictions)

    assert rows[0]["tree_count"] == 4
    assert rows[-1]["tree_count"] == 1
