from hplc.analysis.peak_detection import detect_peaks
import pytest
from hplc.testing.mock_generators import make_raw_signal, expected_peak_properties


def test_detect_peaks_with_ultra_flat_chromatogram_with_no_peaks_finds_no_peaks():
    time, signal = make_raw_signal([], noise_scale=0, duration=600)
    peaks = detect_peaks(time, signal)
    assert len(peaks) == 0

def test_chromatogram_with_no_peaks_finds_no_peaks():
    time, signal = make_raw_signal([], noise_scale=0.1, duration=600)
    peaks = detect_peaks(time, signal)
    assert len(peaks) == 0

@pytest.fixture(scope="module")
def super_easy_chromatogram_peaks():
    peaks_input = [{"mu": 300, "amplitude": 100, "sigma": 12, "alpha": 0.0}]
    time, signal = make_raw_signal(peaks_input, noise_scale=0, duration=600)
    return peaks_input, detect_peaks(time, signal)

def test_detect_peaks_with_no_noise_chromatogram_with_one_peak_finds_one_peak(super_easy_chromatogram_peaks):
    peaks_input, detected_peaks = super_easy_chromatogram_peaks
    assert len(detected_peaks) == 1
    assert abs(detected_peaks[0]["retention_time"] - peaks_input[0]["mu"]) < 5.0

def test_detect_peaks_returns_necessary_attributes(super_easy_chromatogram_peaks):
    peaks_input, detected_peaks = super_easy_chromatogram_peaks
    peak = detected_peaks[0]
    assert "mu" in peak
    assert "amplitude" in peak
    assert "sigma" in peak
    assert "alpha" in peak
    assert "retention_time" in peak
    assert "area" in peak
    assert "start_time" in peak
    assert "end_time" in peak
    assert "max_height" in peak

def test_detect_peaks_returns_correct_attributes_in_simplest_case_very_lenient_test(super_easy_chromatogram_peaks):
    peaks_input, detected_peaks = super_easy_chromatogram_peaks
    expected_properties = expected_peak_properties(peaks_input, n_sigma=3.0)[0]
    peak = detected_peaks[0]
    assert abs(peak["retention_time"] - expected_properties["retention_time"]) < 60.0
    assert abs(peak["max_height"] - expected_properties["max_height"]) / expected_properties["max_height"] < 0.5
    assert abs(peak["area"] - expected_properties["area"]) / expected_properties["area"] < 0.5
    assert abs(peak["start_time"] - expected_properties["start_time"]) < 60.0
    assert abs(peak["end_time"] - expected_properties["end_time"]) < 60.0

def test_detect_peaks_returns_correct_attributes_in_simplest_case_very_more_strict_test(super_easy_chromatogram_peaks):
    peaks_input, detected_peaks = super_easy_chromatogram_peaks
    expected_properties = expected_peak_properties(peaks_input, n_sigma=3.0)[0]
    peak = detected_peaks[0]
    assert abs(peak["retention_time"] - expected_properties["retention_time"]) < 20.0
    assert abs(peak["max_height"] - expected_properties["max_height"]) / expected_properties["max_height"] < 0.15
    assert abs(peak["area"] - expected_properties["area"]) / expected_properties["area"] < 0.15
    assert abs(peak["start_time"] - expected_properties["start_time"]) < 20.0
    assert abs(peak["end_time"] - expected_properties["end_time"]) < 20.0

def test_detect_peaks_with_mildly_noisy_chromatogram_with_one_peak_finds_one_peak():
    peaks_input = [{"mu": 300, "amplitude": 100, "sigma": 12, "alpha": 0.0}]
    time, signal = make_raw_signal(peaks_input, noise_scale=0.05, duration=600)
    peaks = detect_peaks(time, signal)
    assert len(peaks) == 1
    assert abs(peaks[0]["retention_time"] - peaks_input[0]["mu"]) < 5.0

def test_detect_peaks_with_noisy_chromatogram_with_one_peak_finds_one_peak():
    peaks_input = [{"mu": 300, "amplitude": 100, "sigma": 12, "alpha": 0.0}]
    time, signal = make_raw_signal(peaks_input, noise_scale=0.2, duration=600)
    peaks = detect_peaks(time, signal)
    assert len(peaks) == 1
    assert abs(peaks[0]["retention_time"] - peaks_input[0]["mu"]) < 5.0

def test_detect_peaks_with_noisy_chromatogram_with_one_peak_finds_one_peak_and_correct_parameters():
    peaks_input = [{"mu": 300, "amplitude": 100, "sigma": 12, "alpha": 0.0}]
    time, signal = make_raw_signal(peaks_input, noise_scale=0.2, duration=600)
    peaks = detect_peaks(time, signal)
    assert len(peaks) == 1
    assert abs(peaks[0]["retention_time"] - peaks_input[0]["mu"]) < 5.0
