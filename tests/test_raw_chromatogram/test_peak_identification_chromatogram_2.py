from hplc.analysis.peak_detection import detect_peaks
import pytest
from hplc.testing.mock_generators import make_raw_signal, expected_peak_properties

CLIPPED_BROAD_PEAKS = [
    {"mu": 800, "amplitude": 800,  "sigma": 120, "alpha": 30.0},
    {"mu": 200, "amplitude": 1000, "sigma": 12,  "alpha": 0.0},
    {"mu": 350, "amplitude": 600,  "sigma": 12,  "alpha": 5.0},
    {"mu": 480, "amplitude": 900,  "sigma": 12,  "alpha": 3.0},
]
NOISE = 0.2
EXPECTED = expected_peak_properties(CLIPPED_BROAD_PEAKS, n_sigma=3.0)
DURATION = 900

@pytest.fixture(scope="module")
def clipped_broad_detected_peaks():
    time, signal = make_raw_signal(CLIPPED_BROAD_PEAKS, noise_scale=NOISE, duration=DURATION)
    return detect_peaks(time, signal)


def test_clipped_broad_finds_all_peaks(clipped_broad_detected_peaks):
    assert len(clipped_broad_detected_peaks) == 4


@pytest.mark.parametrize("i, label", [
    (0, "extreme fronting"),
    (1, "symmetric"),
    (2, "tailing"),
    (3, "mild tailing"),
])
class TestFindPeakProperties:
    def _match(self, clipped_broad_detected_peaks, i):
        exp = EXPECTED[i]
        return min(clipped_broad_detected_peaks, key=lambda p: abs(p["retention_time"] - exp["retention_time"]))

    def test_retention_time(self, clipped_broad_detected_peaks, i, label):
        peak = self._match(clipped_broad_detected_peaks, i)
        assert abs(peak["retention_time"] - EXPECTED[i]["retention_time"]) < 5.0

    def test_height(self, clipped_broad_detected_peaks, i, label):
        peak = self._match(clipped_broad_detected_peaks, i)
        assert abs(peak["max_height"] - EXPECTED[i]["max_height"]) / EXPECTED[i]["max_height"] < 0.05

    def test_area(self, clipped_broad_detected_peaks, i, label):
        peak = self._match(clipped_broad_detected_peaks, i)
        assert abs(peak["area"] - EXPECTED[i]["area"]) / EXPECTED[i]["area"] < 0.05

    def test_start(self, clipped_broad_detected_peaks, i, label):
        peak = self._match(clipped_broad_detected_peaks, i)
        assert peak["start_time"] < EXPECTED[i]["retention_time"]
        assert abs(peak["start_time"] - EXPECTED[i]["start_time"]) < 10.0

    def test_end(self, clipped_broad_detected_peaks, i, label):
        peak = self._match(clipped_broad_detected_peaks, i)
        assert peak["end_time"] > EXPECTED[i]["retention_time"]
        assert abs(peak["end_time"] - EXPECTED[i]["end_time"]) < 10.0