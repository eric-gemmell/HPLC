from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch
from HPLC.models import Signal2D, SignalStack2D
from HPLC.testing.mock_generators import make_signal2d


PEAKS_A = [
    {"mu": 100, "amplitude": 500, "sigma": 10, "alpha": 0},
    {"mu": 300, "amplitude": 800, "sigma": 15, "alpha": 1.5},
]

PEAKS_B = [
    {"mu": 110, "amplitude": 400, "sigma": 12, "alpha": 0.5},
    {"mu": 310, "amplitude": 700, "sigma": 14, "alpha": 1.0},
]


@pytest.fixture
def signal_a():
    return make_signal2d(PEAKS_A, filename="file_A", detector_name="DAD 254 nm", duration=600, seed=1)


@pytest.fixture
def signal_b():
    return make_signal2d(PEAKS_B, filename="file_B", detector_name="DAD 280 nm", duration=500, seed=2)


@pytest.fixture
def signal_c():
    return make_signal2d(PEAKS_B, filename="file_C", detector_name="DAD 210 nm", duration=500, seed=3)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_stack_stores_signals(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        assert len(stack) == 2
        for signal in  stack.signals:
            assert isinstance(signal, Signal2D)

    def test_initial_delta_is_zero(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        for d in stack.deltas:
            assert d == 0.0

    def test_initial_stretch_factors_are_one(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        for sf in stack.stretch_factors:
            assert sf == 1.0

    def test_stack_works_with_just_one_signal(self, signal_a):
        stack = SignalStack2D([signal_a])
        assert len(stack) == 1

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="at least one"):
            SignalStack2D([])


# ---------------------------------------------------------------------------
# Signal selection
# ---------------------------------------------------------------------------

class TestSignalSelection:

    def test_select_by_index(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=5.0, signal=0)
        assert stack.deltas[0] == 5.0
        assert stack.deltas[1] == 0.0

    def test_select_by_name(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=5.0, filename="file_A")
        assert stack.deltas[0] == 5.0
        assert stack.deltas[1] == 0.0

    def test_select_by_detector(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=5.0, detector_name="DAD 254 nm")
        assert stack.deltas[0] == 5.0
        assert stack.deltas[1] == 0.0

    def test_defaults_to_last_signal(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=5.0)
        assert stack.deltas[0] == 0.0
        assert stack.deltas[1] == 5.0

    def test_invalid_index_raises(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(IndexError):
            stack.shift(delta=5.0, signal=5)

    def test_invalid_name_raises(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(KeyError):
            stack.shift(delta=5.0, filename="nonexistent")

        with pytest.raises(KeyError):
            stack.shift(delta=5.0, detector_name="nonexistent")
    
    def test_select_by_ambiguous_raises(self, signal_a):
        stack = SignalStack2D([signal_a, signal_a])  # duplicate signals with same detector name
        with pytest.raises(ValueError, match="Multiple signals match"):
            stack.shift(delta=5.0, detector_name="DAD 254 nm")
        with pytest.raises(ValueError, match="Multiple signals match"):
            stack.shift(delta=5.0, filename = "file_A")
    


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------

class TestShift:

    def test_shift_returns_new_stack(self, signal_a, signal_b): 
        stack = SignalStack2D([signal_a, signal_b])
        result = stack.shift(delta=10.0, signal=1)
        assert isinstance(result, SignalStack2D)
        assert result is not stack

    def test_shift_with_delta(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=10.0, signal=1)
        assert stack.deltas[1] == 10.0

    def test_shift_with_negative_delta(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=-3.5, signal=0)
        assert stack.deltas[0] == -3.5

    def test_shift_with_from_to(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(from_time=110.0, to_time=100.0, signal=1)
        assert stack.deltas[1] == pytest.approx(-10.0)

    def test_shift_accumulates(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=5.0, signal=1)
        stack.shift(delta=3.0, signal=1)
        assert stack.deltas[1] == pytest.approx(8.0)

    def test_shift_rejects_both_delta_and_from_to(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(ValueError, match="[Cc]annot.*both"):
            stack.shift(delta=5.0, from_time=100.0, to_time=110.0, signal=1)

    def test_shift_rejects_neither_delta_nor_from_to(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(ValueError):
            stack.shift(signal=1)

    def test_shift_rejects_partial_from_to(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(ValueError):
            stack.shift(from_time=100.0, signal=1)


# ---------------------------------------------------------------------------
# Stretch
# ---------------------------------------------------------------------------

class TestStretch:

    def test_stretch_returns_new_stack(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        result = stack.stretch(factor=1.05, signal=1)
        assert isinstance(result, SignalStack2D)
        assert result is not stack

    def test_stretch_updates_factor(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.stretch(factor=1.05, signal=1)
        assert stack.stretch_factors[1] == pytest.approx(1.05)

    def test_stretch_accumulates_multiplicatively(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.stretch(factor=1.1, signal=1)
        stack.stretch(factor=0.9, signal=1)
        assert stack.stretch_factors[1] == pytest.approx(1.1 * 0.9)

    def test_stretch_with_anchor_updates_delta(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        anchor = 200.0
        factor = 1.1
        stack.stretch(factor=factor, anchor=anchor, signal=1)
        # anchor point must stay fixed: new_t = factor * (t - anchor) + anchor
        # offset needed: anchor - factor * anchor = anchor * (1 - factor)
        expected_delta = anchor * (1 - factor)
        assert stack.stretch_factors[1] == pytest.approx(factor)
        assert stack.deltas[1] == pytest.approx(expected_delta)

    def test_stretch_default_anchor_is_zero(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.stretch(factor=2.0, signal=1)
        # anchor=0 means no delta adjustment needed
        assert stack.deltas[1] == 0.0

    def test_stretch_rejects_zero_factor(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(ValueError, match="[Ff]actor"):
            stack.stretch(factor=0.0, signal=1)

    def test_stretch_rejects_negative_factor(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(ValueError, match="[Ff]actor"):
            stack.stretch(factor=-1.0, signal=1)


# ---------------------------------------------------------------------------
# Fit (two-point alignment)
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_new_stack(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        result = stack.fit(
            from_times=(110.0, 310.0),
            to_times=(100.0, 300.0),
            signal=1,
        )
        assert isinstance(result, SignalStack2D)
        assert result is not stack

    def test_fit_computes_correct_stretch_and_delta(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        # We want: t=110 -> 100, t=310 -> 300
        # scale = (300 - 100) / (310 - 110) = 200 / 200 = 1.0
        # offset = 100 - 1.0 * 110 = -10
        stack.fit(
            from_times=(110.0, 310.0),
            to_times=(100.0, 300.0),
            signal=1,
        )
        assert stack.stretch_factors[1] == pytest.approx(1.0)
        assert stack.deltas[1] == pytest.approx(-10.0)

    def test_fit_with_stretch(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        # We want: t=100 -> 100, t=300 -> 330
        # scale = (330 - 100) / (300 - 100) = 230 / 200 = 1.15
        # offset = 100 - 1.15 * 100 = -15
        stack.fit(
            from_times=(100.0, 300.0),
            to_times=(100.0, 330.0),
            signal=1,
        )
        assert stack.stretch_factors[1] == pytest.approx(1.15)
        assert stack.deltas[1] == pytest.approx(-15.0)

    def test_fit_rejects_identical_from_times(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(ValueError, match="[Dd]istinct"):
            stack.fit(
                from_times=(100.0, 100.0),
                to_times=(100.0, 200.0),
                signal=1,
            )

    def test_fit_overwrites_previous_adjustments(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=999.0, signal=1)
        stack.stretch(factor=5.0, signal=1)
        stack.fit(
            from_times=(110.0, 310.0),
            to_times=(100.0, 300.0),
            signal=1,
        )
        assert stack.stretch_factors[1] == pytest.approx(1.0)
        assert stack.deltas[1] == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# Display (smoke test)
# ---------------------------------------------------------------------------

class TestDisplay:

    def test_display_returns_self(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        # We just verify it returns self; actual rendering is a graphing concern
        with patch("HPLC.models.signal_stack.plot_signal_2d"):
            result = stack.display()
        assert result is stack

    def test_display_after_adjustments(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=5.0, signal=1)
        stack.stretch(factor=1.1, signal=0)
        with patch("HPLC.models.signal_stack.plot_signal_2d"):
            result = stack.display()
        assert result is stack

    def test_display_accepts_optional_parameters(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with patch("HPLC.models.signal_stack.plot_signal_2d") as mock_display:
            stack.display(y_offset=50.0, title="My Stack", width=800, height=600, filename="out.png", dpi=150, style="bw", x_lim=(0, 10), y_lim=(0, 100))
            mock_display.assert_called_once()

    def test_display_publication(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with patch("HPLC.models.signal_stack.plot_signal_2d") as mock_display:
            result = stack.display_publication()
            assert result is stack
            mock_display.assert_called_once()

    def test_display_publication_accepts_optional_parameters(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with patch("HPLC.models.signal_stack.plot_signal_2d") as mock_display:
            stack.display_publication(y_offset=50.0, title="My Stack", width=800, height=600, filename="out.png", dpi=150, x_lim=(0, 10), y_lim=(0, 100))
            mock_display.assert_called_once()
