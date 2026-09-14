from src.quality.resolution import resolution_status


def test_good_resolution():
    status, warnings = resolution_status(0.5, 0.5)
    assert status == "good"
    assert warnings == []


def test_blocked_resolution():
    status, warnings = resolution_status(10.0, 10.0)
    assert status == "blocked"
    assert warnings
