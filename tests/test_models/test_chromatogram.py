import pytest
import numpy as np
from unittest.mock import patch

from hplc.models.chromatogram import Chromatogram
from hplc.models.signal import Signal2D
from hplc.graphing.graph_models import GraphProperties2D
from hplc.testing.mock_generators import make_signal2d, make_chromatogram

# ---------------------------------------------------------------------------
# Default display parameters
# ---------------------------------------------------------------------------

DEFAULT_DISPLAY_WIDTH = 540
DEFAULT_DISPLAY_HEIGHT = 400
DEFAULT_DISPLAY_DPI = 300
DEFAULT_DISPLAY_STYLE = "color"

DEFAULT_PUBLICATION_STYLE = "bw"

DEFAULT_STACKED_WIDTH = 1000
DEFAULT_STACKED_HEIGHT = 600
DEFAULT_STACKED_DPI = 300
DEFAULT_STACKED_STYLE = "color"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_signal():
    def _make(detector_name="UV", signal_unit="mAU"):
        return Signal2D(detector_name, np.array([1.0, 2.0, 3.0]), "min", np.array([10.0, 20.0, 30.0]), signal_unit)
    return _make


@pytest.fixture
def chrom(make_signal):
    return Chromatogram(
        signals={"UV": make_signal("UV"), "FLD": make_signal("FLD", "RFU")},
        filename="sample_001.txt",
    )


@pytest.fixture
def rich_chrom():
    """A realistic chromatogram with three detectors and distinct peak profiles."""
    return make_chromatogram(
        signal_definitions={
            "DAD 190 nm": dict(
                peaks=[
                    {"mu": 120, "amplitude": 800, "sigma": 12, "alpha": 1.2},
                    {"mu": 350, "amplitude": 400, "sigma": 18, "alpha": 0.5},
                    {"mu": 600, "amplitude": 600, "sigma": 10, "alpha": 2.0},
                ],
                duration=900, dt=1, noise_scale=0.5, seed=10,
                signal_unit="mAU",
            ),
            "DAD 254 nm": dict(
                peaks=[
                    {"mu": 130, "amplitude": 1200, "sigma": 14, "alpha": 0.8},
                    {"mu": 400, "amplitude": 900, "sigma": 20, "alpha": 1.5},
                ],
                duration=900, dt=1, noise_scale=0.3, seed=20,
                signal_unit="mAU",
            ),
            "FLD": dict(
                peaks=[
                    {"mu": 200, "amplitude": 500, "sigma": 25, "alpha": 0.3},
                    {"mu": 550, "amplitude": 700, "sigma": 15, "alpha": 1.8},
                    {"mu": 750, "amplitude": 300, "sigma": 10, "alpha": 0.0},
                ],
                duration=900, dt=1, noise_scale=0.2, seed=30,
                signal_unit="RFU",
            ),
        },
        filename="caffeine_analysis_042.aia",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction(chrom):
    assert chrom.filename == "sample_001.txt"
    assert set(chrom.signals.keys()) == {"UV", "FLD"}
    assert chrom.additional_data == {}


def test_empty_signals_raises():
    with pytest.raises(ValueError, match="at least one signal"):
        Chromatogram(signals={}, filename="empty.txt")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

@patch("hplc.models.chromatogram.plot_graphs")
def test_display_returns_self(mock_plot, chrom):
    assert chrom.display() is chrom


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_single_detector(mock_plot, chrom):
    chrom.display(detectors=["UV"])
    graphs = mock_plot.call_args.kwargs["graphs"]
    assert len(graphs) == 1


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_multiple_detectors(mock_plot, chrom):
    chrom.display(detectors=["UV", "FLD"])
    graphs = mock_plot.call_args.kwargs["graphs"]
    assert len(graphs) == 2


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_accepts_optional_parameters(mock_plot, chrom):
    chrom.display(width=800, height=600, filename="out.png", dpi=150, style="bw",
                  title="Custom Title", x_min=0, x_max=10, y_min=0, y_max=100)
    mock_plot.assert_called_once()


@patch("hplc.models.chromatogram.plot_graphs")
def test_chromatogram_display_uses_trace_properties_for_each_signal(mock_plot, rich_chrom):
    rich_chrom.display()

    mock_plot.assert_called_once()
    call_kwargs = mock_plot.call_args.kwargs
    graphs = call_kwargs["graphs"]

    assert len(graphs) == 3
    for graph, (name, sig) in zip(graphs, rich_chrom.signals.items()):
        assert isinstance(graph, GraphProperties2D)
        expected_trace = sig.to_trace_properties()
        assert graph.signals == [expected_trace]

@patch("hplc.models.chromatogram.plot_graphs")
def test_chromatogram_display_default_params_are_correct(mock_plot, rich_chrom):
    rich_chrom.display()
    call_kwargs = mock_plot.call_args.kwargs
    for graph_properties in call_kwargs["graphs"]:
        assert graph_properties.x_min == None
        assert graph_properties.x_max == None
        assert graph_properties.y_min == None
        assert graph_properties.y_max == None

    assert call_kwargs["width"] == DEFAULT_DISPLAY_WIDTH
    assert call_kwargs["height"] == DEFAULT_DISPLAY_HEIGHT
    assert call_kwargs["dpi"] == DEFAULT_DISPLAY_DPI
    assert call_kwargs["title"] == rich_chrom.filename
    assert call_kwargs["filename"] == None
    assert call_kwargs["style"] == DEFAULT_DISPLAY_STYLE

@patch("hplc.models.chromatogram.plot_graphs")
def test_chromatogram_display_override_params_are_correct(mock_plot, rich_chrom):
    rich_chrom.display(detectors = ["FLD"],title="the most beautiful chromatogram in the world", filename="savepls.png", x_min=100, x_max=800, dpi=150, y_min=0, y_max=1500, style="bw",width=1000,height=750)
    exp_trace_properties = rich_chrom.signals["FLD"].to_trace_properties()

    call_kwargs = mock_plot.call_args.kwargs
    assert len(call_kwargs["graphs"]) == 1
    graph_properties = call_kwargs["graphs"][0]
    assert graph_properties.x_min == 100
    assert graph_properties.x_max == 800
    assert graph_properties.y_min == 0
    assert graph_properties.y_max == 1500
    obs_trace_properties = graph_properties.signals[0]
    assert obs_trace_properties.name == exp_trace_properties.name
    assert obs_trace_properties.time_unit == exp_trace_properties.time_unit
    assert obs_trace_properties.signal_unit == exp_trace_properties.signal_unit

    assert call_kwargs["title"] == "the most beautiful chromatogram in the world"
    assert call_kwargs["width"] == 1000
    assert call_kwargs["height"] == 750
    assert call_kwargs["dpi"] == 150
    assert call_kwargs["filename"] == "savepls.png"
    assert call_kwargs["style"] == "bw"


# ---------------------------------------------------------------------------
# Display publication
# ---------------------------------------------------------------------------

@patch("hplc.models.chromatogram.plot_graphs")
def test_display_publication_returns_self(mock_plot, chrom):
    assert chrom.display_publication() is chrom
    mock_plot.assert_called_once()


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_publication_single_detector(mock_plot, chrom):
    chrom.display_publication(detectors=["UV"])
    graphs = mock_plot.call_args.kwargs["graphs"]
    assert len(graphs) == 1


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_publication_multiple_detectors(mock_plot, chrom):
    chrom.display_publication(detectors=["UV", "FLD"])
    graphs = mock_plot.call_args.kwargs["graphs"]
    assert len(graphs) == 2


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_publication_accepts_optional_parameters(mock_plot, chrom):
    chrom.display_publication(width=800, height=600, filename="out.png", dpi=150,
                              title="Custom Title", x_min=0, x_max=10, y_min=0, y_max=100)
    mock_plot.assert_called_once()


@patch("hplc.models.chromatogram.plot_graphs")
def test_chromatogram_publication_display_uses_trace_properties_for_each_signal(mock_plot, rich_chrom):
    rich_chrom.display_publication()

    mock_plot.assert_called_once()
    call_kwargs = mock_plot.call_args.kwargs
    graphs = call_kwargs["graphs"]

    assert len(graphs) == 3
    for graph, (name, sig) in zip(graphs, rich_chrom.signals.items()):
        assert isinstance(graph, GraphProperties2D)
        expected_trace = sig.to_trace_properties()
        assert graph.signals == [expected_trace]

@patch("hplc.models.chromatogram.plot_graphs")
def test_chromatogram_publication_display_default_params_are_correct(mock_plot, rich_chrom):
    rich_chrom.display_publication()
    call_kwargs = mock_plot.call_args.kwargs
    for graph_properties in call_kwargs["graphs"]:
        assert graph_properties.x_min == None
        assert graph_properties.x_max == None
        assert graph_properties.y_min == None
        assert graph_properties.y_max == None


    assert call_kwargs["width"] == DEFAULT_DISPLAY_WIDTH
    assert call_kwargs["height"] == DEFAULT_DISPLAY_HEIGHT
    assert call_kwargs["dpi"] == DEFAULT_DISPLAY_DPI
    assert call_kwargs["title"] == rich_chrom.filename
    assert call_kwargs["filename"] == None
    assert call_kwargs["style"] == DEFAULT_PUBLICATION_STYLE

@patch("hplc.models.chromatogram.plot_graphs")
def test_chromatogram_publication_display_override_params_are_correct(mock_plot, rich_chrom):
    rich_chrom.display_publication(detectors = ["FLD"],title="the most beautiful chromatogram in the world", filename="savepls.png", x_min=100, x_max=800, dpi=150, y_min=0, y_max=1500, width=1000,height=750)
    exp_trace_properties = rich_chrom.signals["FLD"].to_trace_properties()

    call_kwargs = mock_plot.call_args.kwargs
    assert len(call_kwargs["graphs"]) == 1
    graph_properties = call_kwargs["graphs"][0]
    assert graph_properties.x_min == 100
    assert graph_properties.x_max == 800
    assert graph_properties.y_min == 0
    assert graph_properties.y_max == 1500
    obs_trace_properties = graph_properties.signals[0]
    assert obs_trace_properties.name == exp_trace_properties.name
    assert obs_trace_properties.time_unit == exp_trace_properties.time_unit
    assert obs_trace_properties.signal_unit == exp_trace_properties.signal_unit

    assert call_kwargs["title"] == "the most beautiful chromatogram in the world"
    assert call_kwargs["width"] == 1000
    assert call_kwargs["height"] == 750
    assert call_kwargs["dpi"] == 150
    assert call_kwargs["filename"] == "savepls.png"
    assert call_kwargs["style"] == "bw"

@patch("hplc.models.chromatogram.plot_graphs")
def test_chromatogram_publication_display_with_two_signals_override_params_are_correct(mock_plot, rich_chrom):
    rich_chrom.display_publication(detectors = ["FLD", "DAD 254 nm"],title="the most beautiful chromatogram in the world", filename="savepls.png", x_min=100, x_max=800, dpi=150, y_min=0, y_max=1500, width=1000,height=750)
    exp_trace_properties = rich_chrom.signals["FLD"].to_trace_properties()

    call_kwargs = mock_plot.call_args.kwargs
    assert len(call_kwargs["graphs"]) == 2
    graph_properties = call_kwargs["graphs"][0]
    assert graph_properties.x_min == 100
    assert graph_properties.x_max == 800
    assert graph_properties.y_min == 0
    assert graph_properties.y_max == 1500
    obs_trace_properties = graph_properties.signals[0]
    assert obs_trace_properties.name == exp_trace_properties.name
    assert obs_trace_properties.time_unit == exp_trace_properties.time_unit
    assert obs_trace_properties.signal_unit == exp_trace_properties.signal_unit

    exp_trace_properties = rich_chrom.signals["DAD 254 nm"].to_trace_properties()

    call_kwargs = mock_plot.call_args.kwargs
    assert len(call_kwargs["graphs"]) == 2
    graph_properties = call_kwargs["graphs"][1]
    assert graph_properties.x_min == 100
    assert graph_properties.x_max == 800
    assert graph_properties.y_min == 0
    assert graph_properties.y_max == 1500
    obs_trace_properties = graph_properties.signals[0]
    assert obs_trace_properties.name == exp_trace_properties.name
    assert obs_trace_properties.time_unit == exp_trace_properties.time_unit
    assert obs_trace_properties.signal_unit == exp_trace_properties.signal_unit

    assert call_kwargs["title"] == "the most beautiful chromatogram in the world"
    assert call_kwargs["width"] == 1000
    assert call_kwargs["height"] == 750
    assert call_kwargs["dpi"] == 150
    assert call_kwargs["filename"] == "savepls.png"
    assert call_kwargs["style"] == "bw"

# ----------------------------------------------------------------------------
# display_stacked
# ----------------------------------------------------------------------------
@patch("hplc.models.chromatogram.plot_2d_graph")
def test_display_stacked_returns_self(mock_plot, chrom):
    assert chrom.display_stacked() is chrom
    mock_plot.assert_called_once()
    signals = mock_plot.call_args.kwargs["signals"]
    assert len(signals) == 2


@patch("hplc.models.chromatogram.plot_2d_graph")
def test_display_stacked_single_detector(mock_plot, chrom):
    chrom.display_stacked(detectors=["UV"])
    signals = mock_plot.call_args.kwargs["signals"]
    assert len(signals) == 1

@patch("hplc.models.chromatogram.plot_2d_graph")
def test_display_stacked_multiple_detectors(mock_plot, chrom):
    chrom.display_stacked(detectors=["UV", "FLD"])
    signals = mock_plot.call_args.kwargs["signals"]
    assert len(signals) == 2

@patch("hplc.models.chromatogram.plot_2d_graph")
def test_display_stacked_accepts_optional_parameters(mock_plot, chrom):
    chrom.display_stacked(width=800, height=600, filename="out.png", dpi=150,
                              title="Custom Title", x_min=0, x_max=10, y_min=0, y_max=100, y_offset = 5)
    mock_plot.assert_called_once()

@patch("hplc.models.chromatogram.plot_2d_graph")
def test_chromatogram_display_stacked_uses_trace_properties_for_each_signal(mock_plot, rich_chrom):
    rich_chrom.display_stacked()

    mock_plot.assert_called_once()

    signals = rich_chrom.signals
    expected_traces = []
    for signal in signals.values():
        trace = signal.to_trace_properties()
        trace.name = f"{signal.detector_name}"
        expected_traces.append(trace)
    mock_plot.assert_called_once_with(
        signals=expected_traces,
        title=rich_chrom.filename,
        width=DEFAULT_STACKED_WIDTH,
        height=DEFAULT_STACKED_HEIGHT,
        filename=None,
        dpi=DEFAULT_STACKED_DPI,
        style=DEFAULT_STACKED_STYLE,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        fig=None,
    )


@patch("hplc.models.chromatogram.plot_2d_graph")
def test_display_stacked_uses_optional_parameters(mock_plot, rich_chrom):
    y_offset = 50.0
    rich_chrom.display_stacked(width=800, height=600, filename="out.png", y_offset=y_offset, dpi=150, style="bw", title="I love my stack", x_min=0, x_max=10, y_min=0, y_max=100)
    mock_plot.assert_called_once()

    
    expected_traces = []
    for i, signal in enumerate(rich_chrom.signals.values()):
        trace = signal.to_trace_properties()
        trace.name = f"{signal.detector_name}"
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
# ---------------------------------------------------------------------------
# Detect peaks
# ---------------------------------------------------------------------------

def test_detect_peaks_returns_chromatogram(chrom):
    result = chrom.detect_peaks()
    assert isinstance(result, Chromatogram)


def test_detect_peaks_returns_copy(chrom):
    result = chrom.detect_peaks()
    assert result is not chrom
    for name in chrom.signals:
        assert result.signals[name] is not chrom.signals[name]


def test_detect_peaks_calls_signal_detect_peaks_once_per_signal(chrom):
    with patch.object(Signal2D, "detect_peaks", return_value=chrom.signals["UV"]) as mock:
        chrom.detect_peaks()
        assert mock.call_count == len(chrom.signals)

