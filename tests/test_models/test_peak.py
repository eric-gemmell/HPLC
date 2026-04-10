import pytest
import numpy as np
from hplc.models.peak import Peak2D, AreaPeak2D, MathPeak2D
from hplc.graphing.graph_models import FillRegion
from hplc.testing.mock_generators import make_raw_signal, expected_peak_properties


# ---------------------------------------------------------------------------
# Shared test data — one signal, two peak representations
# ---------------------------------------------------------------------------

PEAK_DEF = {"mu": 200, "amplitude": 100, "sigma": 12, "alpha": 0.0}

TIME, SIGNAL = make_raw_signal([PEAK_DEF], duration=400, dt=0.1, noise_scale=0.0)
_, BACKGROUND = make_raw_signal([], duration=400, dt=0.1, noise_scale=0.1)
OBSERVED_SIGNAL = SIGNAL + BACKGROUND
PROPS = expected_peak_properties([PEAK_DEF])[0]

MASK = (TIME >= PROPS["start_time"]) & (TIME <= PROPS["end_time"])
EXPECTED_TIME = TIME[MASK]
EXPECTED_OBSERVED_SIGNAL = OBSERVED_SIGNAL[MASK]
EXPECTED_BACKGROUND = BACKGROUND[MASK]

EMPIRICAL_AREA = np.trapezoid(EXPECTED_OBSERVED_SIGNAL - EXPECTED_BACKGROUND, EXPECTED_TIME)
EMPIRICAL_MAX_HEIGHT = np.max(EXPECTED_OBSERVED_SIGNAL - EXPECTED_BACKGROUND)
EMPIRICAL_RETENTION_TIME = EXPECTED_TIME[np.argmax(EXPECTED_OBSERVED_SIGNAL)]

AREA_PEAK = AreaPeak2D(
    start_time=PROPS["start_time"],
    end_time=PROPS["end_time"],
    time=TIME,
    signal=OBSERVED_SIGNAL,
    baseline=BACKGROUND,
)

def test_area_peak_derives_correct_properties():
    fill = AREA_PEAK.to_fill_region()
    assert isinstance(fill, FillRegion)
    assert fill.time[0] == EXPECTED_TIME[0]
    assert fill.time[-1] == EXPECTED_TIME[-1]
    np.testing.assert_array_equal(fill.time, EXPECTED_TIME)
    np.testing.assert_array_equal(fill.upper, EXPECTED_OBSERVED_SIGNAL)
    np.testing.assert_array_equal(fill.lower, EXPECTED_BACKGROUND)

    assert AREA_PEAK.area == pytest.approx(EMPIRICAL_AREA, rel=1e-6)
    assert AREA_PEAK.max_height == pytest.approx(EMPIRICAL_MAX_HEIGHT, rel=1e-6)
    assert AREA_PEAK.retention_time == pytest.approx(EMPIRICAL_RETENTION_TIME, rel=1e-6)

MATH_PEAK = MathPeak2D(
    mu=PEAK_DEF["mu"],
    amplitude=PEAK_DEF["amplitude"],
    sigma=PEAK_DEF["sigma"],
    alpha=PEAK_DEF["alpha"],
    time=TIME,
    baseline=BACKGROUND,
)

def test_math_peak_is_subclass_of_peak2d():
    assert isinstance(MATH_PEAK, Peak2D)

def test_math_peak_derives_correct_properties():
    fill = MATH_PEAK.to_fill_region()
    assert isinstance(fill, FillRegion)
    assert fill.time[0] == EXPECTED_TIME[0]
    assert fill.time[-1] == EXPECTED_TIME[-1]
    np.testing.assert_array_equal(fill.time, EXPECTED_TIME)
    np.testing.assert_array_equal(fill.upper, EXPECTED_OBSERVED_SIGNAL)
    np.testing.assert_array_equal(fill.lower, EXPECTED_BACKGROUND)

    assert MATH_PEAK.area == pytest.approx(PROPS["area"], rel=1e-3)
    assert MATH_PEAK.max_height == pytest.approx(PROPS["max_height"], rel=1e-3)
    assert MATH_PEAK.retention_time == pytest.approx(PROPS["retention_time"], rel=1e-3)
    assert MATH_PEAK.start_time == pytest.approx(PROPS["start_time"], rel=1e-3)
    assert MATH_PEAK.end_time == pytest.approx(PROPS["end_time"], rel=1e-3)

    assert MATH_PEAK.mu == PEAK_DEF["mu"]
    assert MATH_PEAK.amplitude == PEAK_DEF["amplitude"]
    assert MATH_PEAK.sigma == PEAK_DEF["sigma"]
    assert MATH_PEAK.alpha == PEAK_DEF["alpha"]


def test_area_peak_works_without_baseline():
    peak = AreaPeak2D(
        start_time=PROPS["start_time"],
        end_time=PROPS["end_time"],
        time=TIME,
        signal=OBSERVED_SIGNAL,
    )
    assert peak.baseline is None
    assert peak.area is not None
    assert peak.max_height is not None
    assert peak.retention_time is not None

    fill = peak.to_fill_region()
    assert isinstance(fill, FillRegion)
    np.testing.assert_array_equal(fill.upper, EXPECTED_OBSERVED_SIGNAL)
    np.testing.assert_array_equal(fill.lower, np.zeros_like(EXPECTED_OBSERVED_SIGNAL))


def test_math_peak_works_without_baseline():
    peak = MathPeak2D(
        mu=PEAK_DEF["mu"],
        amplitude=PEAK_DEF["amplitude"],
        sigma=PEAK_DEF["sigma"],
        alpha=PEAK_DEF["alpha"],
        time=TIME,
    )
    assert peak.baseline is None
    assert peak.area is not None
    assert peak.max_height is not None
    assert peak.retention_time is not None

    fill = peak.to_fill_region()
    assert isinstance(fill, FillRegion)
    np.testing.assert_array_equal(fill.upper, SIGNAL[MASK])
    np.testing.assert_array_equal(fill.lower, np.zeros_like(SIGNAL[MASK]))


# ---------------------------------------------------------------------------
# Recalculate tests
# ---------------------------------------------------------------------------
_, NEW_BACKGROUND = make_raw_signal([], duration=400, dt=0.1, noise_scale=0.2)
NEW_OBSERVED_SIGNAL = SIGNAL + NEW_BACKGROUND

NEW_EXPECTED_OBSERVED_SIGNAL = NEW_OBSERVED_SIGNAL[MASK]
NEW_EXPECTED_BACKGROUND = NEW_BACKGROUND[MASK]
NEW_EMPIRICAL_AREA = np.trapezoid(NEW_EXPECTED_OBSERVED_SIGNAL - NEW_EXPECTED_BACKGROUND, EXPECTED_TIME)
NEW_EMPIRICAL_MAX_HEIGHT = np.max(NEW_EXPECTED_OBSERVED_SIGNAL - NEW_EXPECTED_BACKGROUND)
NEW_EMPIRICAL_RETENTION_TIME = EXPECTED_TIME[np.argmax(NEW_EXPECTED_OBSERVED_SIGNAL)]


def test_area_peak_has_recalculate_method():
    peak = Peak2D(
        start_time=PROPS["start_time"],
        end_time=PROPS["end_time"],
        time=TIME,
        signal=OBSERVED_SIGNAL,
        baseline=BACKGROUND,
    )
    peak.recalculate(signal=NEW_OBSERVED_SIGNAL, baseline=NEW_BACKGROUND)

    fill = peak.to_fill_region()
    np.testing.assert_array_equal(fill.time, EXPECTED_TIME)
    np.testing.assert_array_equal(fill.upper, NEW_EXPECTED_OBSERVED_SIGNAL)
    np.testing.assert_array_equal(fill.lower, NEW_EXPECTED_BACKGROUND)

    assert peak.area == pytest.approx(NEW_EMPIRICAL_AREA, rel=1e-6)
    assert peak.max_height == pytest.approx(NEW_EMPIRICAL_MAX_HEIGHT, rel=1e-6)
    assert peak.retention_time == pytest.approx(NEW_EMPIRICAL_RETENTION_TIME, rel=1e-6)

    # start/end times do not change
    assert peak.start_time == pytest.approx(PROPS["start_time"], rel=1e-6)
    assert peak.end_time == pytest.approx(PROPS["end_time"], rel=1e-6)


def test_math_peak_has_recalculate_method():
    peak = MathPeak2D(
        mu=PEAK_DEF["mu"],
        amplitude=PEAK_DEF["amplitude"],
        sigma=PEAK_DEF["sigma"],
        alpha=PEAK_DEF["alpha"],
        time=TIME,
        baseline=BACKGROUND,
    )
    peak.recalculate(baseline=NEW_BACKGROUND)

    fill = peak.to_fill_region()
    np.testing.assert_array_equal(fill.time, EXPECTED_TIME)
    np.testing.assert_array_equal(fill.upper, SIGNAL[MASK] + NEW_EXPECTED_BACKGROUND)
    np.testing.assert_array_equal(fill.lower, NEW_EXPECTED_BACKGROUND)

    # Peak shape doesn't change — area, max_height, retention_time, start/end all stay the same
    assert peak.area == pytest.approx(PROPS["area"], rel=1e-3)
    assert peak.max_height == pytest.approx(PROPS["max_height"], rel=1e-3)
    assert peak.retention_time == pytest.approx(PROPS["retention_time"], rel=1e-3)
    assert peak.start_time == pytest.approx(PROPS["start_time"], rel=1e-3)
    assert peak.end_time == pytest.approx(PROPS["end_time"], rel=1e-3)

    # Math parameters unchanged
    assert peak.mu == PEAK_DEF["mu"]
    assert peak.amplitude == PEAK_DEF["amplitude"]
    assert peak.sigma == PEAK_DEF["sigma"]
    assert peak.alpha == PEAK_DEF["alpha"]


def test_math_peak_recalculate_ignores_signal():
    peak = MathPeak2D(
        mu=PEAK_DEF["mu"],
        amplitude=PEAK_DEF["amplitude"],
        sigma=PEAK_DEF["sigma"],
        alpha=PEAK_DEF["alpha"],
        time=TIME,
        baseline=BACKGROUND,
    )
    peak.recalculate(signal=np.zeros_like(TIME), baseline=NEW_BACKGROUND)
    fill = peak.to_fill_region()
    np.testing.assert_array_equal(fill.upper, SIGNAL[MASK] + NEW_EXPECTED_BACKGROUND)