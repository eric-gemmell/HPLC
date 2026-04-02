from hplc.analysis.peak_detection import detect_peaks
import pytest
from hplc.testing.mock_generators import make_raw_signal, expected_peak_properties


OVERLAP_EASY_PEAKS = [
    {"mu": 100, "amplitude": 800,  "sigma": 120, "alpha": 30.0},
    {"mu": 200, "amplitude": 1000, "sigma": 12,  "alpha": 0.0},
    {"mu": 350, "amplitude": 600,  "sigma": 12,  "alpha": 5.0},
    {"mu": 480, "amplitude": 900,  "sigma": 12,  "alpha": 3.0},
    {"mu": 600, "amplitude": 900,  "sigma": 10,  "alpha": 0.0},
    {"mu": 640, "amplitude": 200,  "sigma": 10,  "alpha": 0.0},
]

NOISE = 0.1
DURATION = 1200
EXPECTED = expected_peak_properties(OVERLAP_EASY_PEAKS, n_sigma=3.0)


@pytest.fixture(scope="module")
def detected_peaks():
    time, signal = make_raw_signal(OVERLAP_EASY_PEAKS, noise_scale=NOISE, duration=DURATION)
    return detect_peaks(time, signal)


def test_finds_correct_number_of_peaks(detected_peaks):
    assert len(detected_peaks) == 6


@pytest.mark.parametrize("i, label", [
    (0, "peak 0"),
    (1, "peak 1"),
    (2, "peak 2"),
    (3, "peak 3"),
    (4, "peak 4"),
    (5, "peak 5"),
])
class TestFindPeakProperties:
    def _match(self, detected_peaks, i):
        exp = EXPECTED[i]
        return min(detected_peaks, key=lambda p: abs(p["retention_time"] - exp["retention_time"]))

    def test_retention_time(self, detected_peaks, i, label):
        peak = self._match(detected_peaks, i)
        assert abs(peak["retention_time"] - EXPECTED[i]["retention_time"]) < 5.0

    def test_height(self, detected_peaks, i, label):
        peak = self._match(detected_peaks, i)
        assert abs(peak["max_height"] - EXPECTED[i]["max_height"]) / EXPECTED[i]["max_height"] < 0.05

    def test_area(self, detected_peaks, i, label):
        peak = self._match(detected_peaks, i)
        assert abs(peak["area"] - EXPECTED[i]["area"]) / EXPECTED[i]["area"] < 0.05

    def test_start(self, detected_peaks, i, label):
        peak = self._match(detected_peaks, i)
        assert peak["start_time"] < EXPECTED[i]["retention_time"]
        assert abs(peak["start_time"] - EXPECTED[i]["start_time"]) < 10.0

    def test_end(self, detected_peaks, i, label):
        peak = self._match(detected_peaks, i)
        assert peak["end_time"] > EXPECTED[i]["retention_time"]
        assert abs(peak["end_time"] - EXPECTED[i]["end_time"]) < 10.0