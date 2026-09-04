from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch
from hplc.models import Signal2D, SignalStack2D
from hplc.testing.mock_generators import make_signal2d


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

    def test_initial_y_scales_are_one(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        for ys in stack.y_scales:
            assert ys == 1.0

    def test_initial_labels_are_none(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        for label in stack.labels:
            assert label is None

    def test_initial_colors_are_none(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        for color in stack.colors:
            assert color is None

    def test_initial_line_widths_are_none(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        for line_width in stack.line_widths:
            assert line_width is None

    def test_stack_works_with_just_one_signal(self, signal_a):
        stack = SignalStack2D([signal_a])
        assert len(stack) == 1

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="at least one"):
            SignalStack2D([])
    
    def test_default_title(self, signal_a):
        stack = SignalStack2D([signal_a])
        assert stack.title == "signal stack"

    def test_custom_title(self, signal_a):
        stack = SignalStack2D([signal_a], title="My Stack")
        assert stack.title == "My Stack"


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
# Scale Y (per-signal vertical scaling)
# ---------------------------------------------------------------------------

class TestScaleY:

    def test_scale_y_returns_new_stack(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        result = stack.scale_y(factor=1.5, signal=1)
        assert isinstance(result, SignalStack2D)
        assert result is not stack

    def test_scale_y_updates_factor(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.scale_y(factor=1.5, signal=1)
        assert stack.y_scales[1] == pytest.approx(1.5)
        assert stack.y_scales[0] == 1.0

    def test_scale_y_accumulates_multiplicatively(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.scale_y(factor=1.1, signal=1)
        stack.scale_y(factor=0.9, signal=1)
        assert stack.y_scales[1] == pytest.approx(1.1 * 0.9)

    def test_scale_y_rejects_zero_factor(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(ValueError, match="[Ff]actor"):
            stack.scale_y(factor=0.0, signal=1)

    def test_scale_y_rejects_negative_factor(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(ValueError, match="[Ff]actor"):
            stack.scale_y(factor=-1.0, signal=1)

    def test_scale_y_select_by_name(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.scale_y(factor=2.0, filename="file_A")
        assert stack.y_scales[0] == pytest.approx(2.0)
        assert stack.y_scales[1] == 1.0

    def test_scale_y_select_by_detector(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.scale_y(factor=2.0, detector_name="DAD 280 nm")
        assert stack.y_scales[0] == 1.0
        assert stack.y_scales[1] == pytest.approx(2.0)

    def test_scale_y_defaults_to_last_signal(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.scale_y(factor=2.0)
        assert stack.y_scales[0] == 1.0
        assert stack.y_scales[1] == pytest.approx(2.0)

    def test_scale_y_invalid_index_raises(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with pytest.raises(IndexError):
            stack.scale_y(factor=2.0, signal=5)


# ---------------------------------------------------------------------------
# Set label
# ---------------------------------------------------------------------------

class TestSetLabel:

    def test_set_label_updates_label(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.set_label(1, "My Custom Label")
        assert stack.labels[1] == "My Custom Label"
        assert stack.labels[0] is None

    def test_set_label_returns_new_stack(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        result = stack.set_label(0, "My Custom Label")
        assert isinstance(result, SignalStack2D)
        assert result is not stack


# ---------------------------------------------------------------------------
# Set color
# ---------------------------------------------------------------------------

class TestSetColor:

    def test_set_color_updates_color(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.set_color(1, "#ff0000")
        assert stack.colors[1] == "#ff0000"
        assert stack.colors[0] is None

    def test_set_color_returns_new_stack(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        result = stack.set_color(0, "#ff0000")
        assert isinstance(result, SignalStack2D)
        assert result is not stack


# ---------------------------------------------------------------------------
# Set line width
# ---------------------------------------------------------------------------

class TestSetLineWidth:

    def test_set_line_width_updates_line_width(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.set_line_width(1, 4.0)
        assert stack.line_widths[1] == 4.0
        assert stack.line_widths[0] is None

    def test_set_line_width_returns_new_stack(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        result = stack.set_line_width(0, 4.0)
        assert isinstance(result, SignalStack2D)
        assert result is not stack


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
        with patch("hplc.models.signal_stack.plot_2d_graph"):
            result = stack.display()
        assert result is stack

    def test_display_after_adjustments(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        stack.shift(delta=5.0, signal=1)
        stack.stretch(factor=1.1, signal=0)
        with patch("hplc.models.signal_stack.plot_2d_graph"):
            result = stack.display()
        assert result is stack

    def test_display_accepts_optional_parameters(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with patch("hplc.models.signal_stack.plot_2d_graph") as mock_display:
            stack.display(y_offset=50.0, title="My Stack", width=800, height=600, filename="out.png", dpi=150, style="bw", x_min=0, x_max=10, y_min=0, y_max=100)
            mock_display.assert_called_once()

    def test_display_publication(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with patch("hplc.models.signal_stack.plot_2d_graph") as mock_display:
            result = stack.display_publication()
            assert result is stack
            mock_display.assert_called_once()

    def test_display_publication_accepts_optional_parameters(self, signal_a, signal_b):
        stack = SignalStack2D([signal_a, signal_b])
        with patch("hplc.models.signal_stack.plot_2d_graph") as mock_display:
            stack.display_publication(y_offset=50.0, title="My Stack", width=800, height=600, filename="out.png", dpi=150, x_min=0, x_max=10, y_min=0, y_max=100)
            mock_display.assert_called_once()


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_to_trace_properties_calls_the_signal_methods(mock_plot, signal_a, signal_b, signal_c):
    stack = SignalStack2D([signal_a, signal_b, signal_c])
    stack.display()

    expected_traces = [
        signal_a.to_trace_properties(),
        signal_b.to_trace_properties(),
        signal_c.to_trace_properties(),
    ]

    mock_plot.assert_called_once_with(
        signals=expected_traces,
        title=stack.title,
        width=1000,
        height=600,
        filename=None,
        dpi=300,
        style="color",
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        fig=None,
    )

@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_accepts_optional_parameters(mock_plot, signal_a, signal_b, signal_c):
    signals = [signal_a, signal_b, signal_c]
    stack = SignalStack2D(signals)
    y_offset = 50.0
    stack.display(width=800, height=600, filename="out.png", y_offset=y_offset, dpi=150, style="bw", title="I love my stack", x_min=0, x_max=10, y_min=0, y_max=100)

    expected_traces = []
    for i, signal in enumerate(signals):
        trace = signal.to_trace_properties()
        trace.signal += y_offset * i
        if trace.baseline is not None:
            trace.baseline += y_offset * i

        for fill in trace.fills:
            fill.upper += y_offset * i
            fill.lower += y_offset * i
        expected_traces.append(trace)

    mock_plot.assert_called_once_with(
        signals=expected_traces,
        title="I love my stack",
        width=800,
        height=600,
        filename="out.png",
        dpi=150,
        style="bw",
        x_min=0,
        x_max=10,
        y_min=0,
        y_max=100,
        fig=None,
    )

@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_applies_y_scale_to_plotted_signal(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    stack = stack.scale_y(factor=2.0, signal=1)
    assert stack.y_scales == [1.0, 2.0]

    stack.display()

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    expected_signal_1 = signal_b.to_trace_properties().signal * 2.0
    np.testing.assert_allclose(plotted_traces[1].signal, expected_signal_1)
    np.testing.assert_allclose(plotted_traces[0].signal, signal_a.to_trace_properties().signal)


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_uses_custom_label(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    stack = stack.set_label(1, "My Custom Label")

    stack.display()

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[1].name == "My Custom Label"
    assert plotted_traces[0].name == signal_a.to_trace_properties().name


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_accepts_labels_array_overriding_all_defaults(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])

    stack.display(labels=["Label A", "Label B"])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].name == "Label A"
    assert plotted_traces[1].name == "Label B"


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_labels_array_overrides_previously_set_label(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    stack = stack.set_label(1, "My Custom Label")

    stack.display(labels=["Label A", "Label B"])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].name == "Label A"
    assert plotted_traces[1].name == "Label B"


def test_display_rejects_labels_array_of_wrong_length(signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    with pytest.raises(ValueError, match="labels"):
        stack.display(labels=["Only One Label"])


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_publication_accepts_labels_array_overriding_all_defaults(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])

    stack.display_publication(labels=["Label A", "Label B"])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].name == "Label A"
    assert plotted_traces[1].name == "Label B"


def test_display_publication_rejects_labels_array_of_wrong_length(signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    with pytest.raises(ValueError, match="labels"):
        stack.display_publication(labels=["Only One Label"])


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_uses_custom_color(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    stack = stack.set_color(1, "#ff0000")

    stack.display()

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[1].color == "#ff0000"
    assert plotted_traces[0].color is None


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_accepts_colors_array_overriding_all_defaults(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])

    stack.display(colors=["#ff0000", "#00ff00"])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].color == "#ff0000"
    assert plotted_traces[1].color == "#00ff00"


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_colors_array_overrides_previously_set_color(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    stack = stack.set_color(1, "#000000")

    stack.display(colors=["#ff0000", "#00ff00"])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].color == "#ff0000"
    assert plotted_traces[1].color == "#00ff00"


def test_display_rejects_colors_array_of_wrong_length(signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    with pytest.raises(ValueError, match="colors"):
        stack.display(colors=["#ff0000"])


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_publication_accepts_colors_array_overriding_all_defaults(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])

    stack.display_publication(colors=["#ff0000", "#00ff00"])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].color == "#ff0000"
    assert plotted_traces[1].color == "#00ff00"


def test_display_publication_rejects_colors_array_of_wrong_length(signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    with pytest.raises(ValueError, match="colors"):
        stack.display_publication(colors=["#ff0000"])


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_uses_custom_line_width(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    stack = stack.set_line_width(1, 4.0)

    stack.display()

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[1].line_width == 4.0
    assert plotted_traces[0].line_width is None


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_accepts_line_widths_array_overriding_all_defaults(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])

    stack.display(line_widths=[2.0, 4.0])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].line_width == 2.0
    assert plotted_traces[1].line_width == 4.0


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_line_widths_array_overrides_previously_set_line_width(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    stack = stack.set_line_width(1, 1.0)

    stack.display(line_widths=[2.0, 4.0])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].line_width == 2.0
    assert plotted_traces[1].line_width == 4.0


def test_display_rejects_line_widths_array_of_wrong_length(signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    with pytest.raises(ValueError, match="line_widths"):
        stack.display(line_widths=[2.0])


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_publication_accepts_line_widths_array_overriding_all_defaults(mock_plot, signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])

    stack.display_publication(line_widths=[2.0, 4.0])

    plotted_traces = mock_plot.call_args.kwargs["signals"]
    assert plotted_traces[0].line_width == 2.0
    assert plotted_traces[1].line_width == 4.0


def test_display_publication_rejects_line_widths_array_of_wrong_length(signal_a, signal_b):
    stack = SignalStack2D([signal_a, signal_b])
    with pytest.raises(ValueError, match="line_widths"):
        stack.display_publication(line_widths=[2.0])


@patch("hplc.models.signal_stack.plot_2d_graph")
def test_display_shifts_peaks_according_to_previously_set_stretch_and_fit_optional_parameters(mock_plot, signal_a, signal_b, signal_c):
    signals = [signal_a, signal_b, signal_c]
    stack = SignalStack2D(signals)
    stack = stack.shift(delta=5.0, signal=0).stretch(factor=2, signal=1).fit(from_times=(90.0, 310.0), to_times=(100.0, 300.0), signal=2)
    y_offset = 50.0
    stack.display(width=800, height=600, filename="out.png", y_offset=y_offset, dpi=150, style="bw", title="I love my stack", x_min=0, x_max=10, y_min=0, y_max=100)

    offset_traces = []
    for i, signal in enumerate(signals):
        trace = signal.to_trace_properties()

        trace.signal += y_offset * i
        if trace.baseline is not None:
            trace.baseline += y_offset * i

        for fill in trace.fills:
            fill.upper += y_offset * i
            fill.lower += y_offset * i

        offset_traces.append(trace)

    expected_traces = []
    for i, trace in enumerate(offset_traces):
        if i == 0:
            trace.time += 5.0
            for fill in trace.fills:
                fill.time += 5.0

        elif i == 1:
            trace.time *= 2

        elif i == 2:
            stretch_factor =  (300.0 - 100.0) / (310.0-90.0) 
            delta = 100.0 - stretch_factor * 90.0
            trace.time = trace.time * stretch_factor + delta

        expected_traces.append(trace)

    mock_plot.assert_called_once_with(
        signals=expected_traces,
        title="I love my stack",
        width=800,
        height=600,
        filename="out.png",
        dpi=150,
        style="bw",
        x_min=0,
        x_max=10,
        y_min=0,
        y_max=100,
        fig=None,
    )