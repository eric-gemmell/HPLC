from hplc.analysis.peak_detection import detect_peaks
import pytest
from hplc.testing.mock_generators import make_raw_signal, expected_peak_properties

DIFFICULT_PEAKS = [
    {"mu": 800, "amplitude": 80,  "sigma": 8,  "alpha": -10.0},  # extreme fronting
    {"mu": 200, "amplitude": 100, "sigma": 12, "alpha": 0.0},    # symmetric
    {"mu": 350, "amplitude": 60,  "sigma": 12, "alpha": 5.0},    # tailing
    {"mu": 480, "amplitude": 90,  "sigma": 12, "alpha": 3.0},    # mild tailing
]
NOISE = 0.1

EXPECTED = expected_peak_properties(DIFFICULT_PEAKS, n_sigma=3.0)


@pytest.fixture(scope="module")
def detected_peaks():
    time, signal = make_raw_signal(DIFFICULT_PEAKS, noise_scale=NOISE, duration=1200)
    return detect_peaks(time, signal)


def test_finds_correct_number_of_peaks(detected_peaks):
    assert len(detected_peaks) == 4


@pytest.mark.parametrize("i, label", [
    (0, "extreme fronting"),
    (1, "symmetric"),
    (2, "tailing"),
    (3, "mild tailing"),
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