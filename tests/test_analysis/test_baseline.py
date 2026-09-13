import numpy as np

from hplc.analysis.baseline import compute_baseline
from hplc.testing.mock_generators import make_raw_signal


PEAK_DEFS = [
    {"mu": 100, "amplitude": 100, "sigma": 8, "alpha": 0.0},
    {"mu": 250, "amplitude": 60, "sigma": 10, "alpha": 0.0},
]

TIME, SIGNAL = make_raw_signal(PEAK_DEFS, duration=400, dt=0.1, noise_scale=2.0)


def test_compute_baseline_returns_array_of_the_same_size():
    baseline = compute_baseline(TIME, SIGNAL)

    assert isinstance(baseline, np.ndarray)
    assert baseline.shape == SIGNAL.shape
    assert np.isfinite(baseline).all()


def test_compute_baseline_is_deterministic():
    np.testing.assert_array_equal(
        compute_baseline(TIME, SIGNAL), compute_baseline(TIME, SIGNAL)
    )
