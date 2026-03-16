import numpy as np
import pytest

from hplc import Signal2D


@pytest.fixture
def mock_elsd_signal():
    """Mock ELSD signal: sampled every 0.1s, baseline 3 ± 1 mV."""
    duration = 10  # seconds
    sampling_interval = 0.1  # every 0.1 second
    time = np.arange(0, duration, sampling_interval)
    baseline = 3.0
    noise = np.random.default_rng(42).uniform(-1, 1, len(time))
    signal = baseline + noise
    return Signal2D(
        time=time, signal=signal, detector_name="ELSD",
        time_unit="mins", measurement_unit="mV",
    )


class TestSignal2DConstruction:
    def test_creates_with_valid_data(self):
        time = np.array([0.0, 0.1, 0.2])
        signal = np.array([10.0, 20.0, 30.0])
        sig = Signal2D(
            time=time, signal=signal, detector_name="UV",
            time_unit="mins", measurement_unit="mAU",
            metadata={"units": "mAU"},
        )

        assert sig.detector_name == "UV"
        assert sig.time_unit == "mins"
        assert sig.measurement_unit == "mAU"
        np.testing.assert_array_equal(sig.time, time)
        np.testing.assert_array_equal(sig.signal, signal)
        assert sig.metadata["units"] == "mAU"

    def test_metadata_optional(self):
        sig = Signal2D(
            time=np.array([0.0, 0.1]),
            signal=np.array([1.0, 2.0]),
            detector_name="ELSD",
            time_unit="mins",
            measurement_unit="mV",
        )
        assert sig.metadata == {}

    def test_rejects_mismatched_lengths(self):
        with pytest.raises((ValueError, TypeError)):
            Signal2D(
                time=np.array([0.0, 0.1, 0.2]),
                signal=np.array([10.0, 20.0]),
                detector_name="UV",
                time_unit="mins",
                measurement_unit="mAU",
            )
