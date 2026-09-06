import numpy as np
import pytest
from unittest.mock import patch

from hplc.models import Chromatogram, Signal2D, Signal3D
from hplc.graphing.graph_models import GraphProperties2D, GraphProperties3D


TIME = [0.0, 1.0, 2.0]
WAVELENGTHS = [200.0, 210.0, 220.0, 230.0]
CUBE = [
    [0.0, 1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0, 7.0],
    [8.0, 9.0, 10.0, 11.0],
]


def make_pda(detector_name="PDA"):
    return Signal3D(
        filename="test.lcd",
        detector_name=detector_name,
        time=TIME,
        time_unit="s",
        wavelengths=WAVELENGTHS,
        wavelength_unit="nm",
        signal=CUBE,
        signal_unit="mAU",
    )


def make_rid(detector_name="RID"):
    return Signal2D(
        filename="test.lcd",
        detector_name=detector_name,
        time=TIME,
        time_unit="s",
        signal=[1.0, 2.0, 3.0],
        signal_unit="uRI",
    )


@pytest.fixture
def chrom_3d():
    return Chromatogram(signals={"PDA": make_pda()}, filename="test.lcd")


@pytest.fixture
def chrom_mixed():
    return Chromatogram(
        signals={"RID": make_rid(), "PDA": make_pda()}, filename="test.lcd"
    )


@pytest.fixture
def chrom_2d():
    return Chromatogram(signals={"RID": make_rid()}, filename="test.lcd")


# ---------------------------------------------------------------------------
# to_graph_properties()
# ---------------------------------------------------------------------------

def test_to_graph_properties_returns_a_panel_per_signal(chrom_mixed):
    graphs = chrom_mixed.to_graph_properties()
    assert len(graphs) == 2


def test_to_graph_properties_uses_2d_panel_for_2d_signal(chrom_2d):
    graphs = chrom_2d.to_graph_properties()
    assert isinstance(graphs[0], GraphProperties2D)
    assert graphs[0].signals == [chrom_2d.signals["RID"].to_trace_properties()]


def test_to_graph_properties_uses_3d_panel_for_3d_signal(chrom_3d):
    graphs = chrom_3d.to_graph_properties()
    assert isinstance(graphs[0], GraphProperties3D)
    assert graphs[0].signal == chrom_3d.signals["PDA"].to_trace_properties()


def test_to_graph_properties_dispatches_per_signal(chrom_mixed):
    graphs = chrom_mixed.to_graph_properties()
    kinds = {g.title: type(g) for g in graphs}
    assert kinds["RID"] is GraphProperties2D
    assert kinds["PDA"] is GraphProperties3D


def test_to_graph_properties_titles_are_the_signal_keys(chrom_mixed):
    titles = [g.title for g in chrom_mixed.to_graph_properties()]
    assert sorted(titles) == ["PDA", "RID"]


def test_to_graph_properties_passes_limits_through(chrom_mixed):
    graphs = chrom_mixed.to_graph_properties(
        x_min=0, x_max=2, y_min=1, y_max=9, z_min=0, z_max=11
    )
    for g in graphs:
        assert (g.x_min, g.x_max) == (0, 2)
        assert (g.y_min, g.y_max) == (1, 9)
    panel_3d = next(g for g in graphs if isinstance(g, GraphProperties3D))
    assert (panel_3d.z_min, panel_3d.z_max) == (0, 11)


def test_to_graph_properties_honours_detector_selection(chrom_mixed):
    graphs = chrom_mixed.to_graph_properties(detectors=["PDA"])
    assert len(graphs) == 1
    assert isinstance(graphs[0], GraphProperties3D)


# ---------------------------------------------------------------------------
# display() must not break on 3D signals
# ---------------------------------------------------------------------------

@patch("hplc.models.chromatogram.plot_graphs")
def test_display_returns_self_for_3d(mock_plot, chrom_3d):
    assert chrom_3d.display() is chrom_3d


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_passes_3d_panel_to_plot_graphs(mock_plot, chrom_3d):
    chrom_3d.display()
    graphs = mock_plot.call_args.kwargs["graphs"]
    assert isinstance(graphs[0], GraphProperties3D)


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_handles_mixed_chromatogram(mock_plot, chrom_mixed):
    chrom_mixed.display()
    graphs = mock_plot.call_args.kwargs["graphs"]
    assert {type(g) for g in graphs} == {GraphProperties2D, GraphProperties3D}


@patch("hplc.models.chromatogram.plot_graphs")
def test_display_publication_handles_3d(mock_plot, chrom_mixed):
    assert chrom_mixed.display_publication() is chrom_mixed
    mock_plot.assert_called_once()


# ---------------------------------------------------------------------------
# display_stacked() only makes sense for 2D signals
# ---------------------------------------------------------------------------

@patch("hplc.models.chromatogram.plot_2d_graph")
def test_display_stacked_skips_3d_signals(mock_plot, chrom_mixed):
    chrom_mixed.display_stacked()
    traces = mock_plot.call_args.kwargs["signals"]
    assert len(traces) == 1
    assert traces[0].name == "RID"


def test_display_stacked_raises_when_no_2d_signals(chrom_3d):
    with pytest.raises(ValueError, match="2D"):
        chrom_3d.display_stacked()


# ---------------------------------------------------------------------------
# detect_peaks() must not break on 3D signals
# ---------------------------------------------------------------------------

def test_detect_peaks_skips_3d_signals(chrom_mixed):
    result = chrom_mixed.detect_peaks()
    assert isinstance(result, Chromatogram)
    assert isinstance(result.signals["PDA"], Signal3D)


def test_detect_peaks_still_processes_2d_signals(chrom_mixed):
    result = chrom_mixed.detect_peaks()
    assert isinstance(result.signals["RID"], Signal2D)


# ---------------------------------------------------------------------------
# extract_wavelengths()
# ---------------------------------------------------------------------------

def test_extract_wavelengths_returns_a_chromatogram(chrom_3d):
    assert isinstance(chrom_3d.extract_wavelengths(210), Chromatogram)


def test_extract_wavelengths_returns_a_copy(chrom_3d):
    result = chrom_3d.extract_wavelengths(210)
    assert result is not chrom_3d
    assert "PDA 210 nm" not in chrom_3d.signals


def test_extract_wavelengths_adds_a_2d_signal(chrom_3d):
    result = chrom_3d.extract_wavelengths(210)
    assert "PDA 210 nm" in result.signals
    extracted = result.signals["PDA 210 nm"]
    assert isinstance(extracted, Signal2D)
    np.testing.assert_array_equal(extracted.signal, [1.0, 5.0, 9.0])


def test_extract_wavelengths_keeps_the_original_signals(chrom_mixed):
    result = chrom_mixed.extract_wavelengths(210)
    assert "PDA" in result.signals
    assert "RID" in result.signals


def test_extract_wavelengths_accepts_a_list(chrom_3d):
    result = chrom_3d.extract_wavelengths([200, 220])
    assert "PDA 200 nm" in result.signals
    assert "PDA 220 nm" in result.signals
    np.testing.assert_array_equal(result.signals["PDA 200 nm"].signal, [0.0, 4.0, 8.0])
    np.testing.assert_array_equal(result.signals["PDA 220 nm"].signal, [2.0, 6.0, 10.0])


def test_extract_wavelengths_snaps_to_nearest(chrom_3d):
    result = chrom_3d.extract_wavelengths(214)
    assert "PDA 210 nm" in result.signals


def test_extract_wavelengths_passes_bandwidth_through(chrom_3d):
    result = chrom_3d.extract_wavelengths(215, bandwidth=20)
    extracted = next(iter(v for k, v in result.signals.items() if k != "PDA"))
    np.testing.assert_allclose(extracted.signal, [1.5, 5.5, 9.5])


def test_extract_wavelengths_rejects_out_of_range(chrom_3d):
    with pytest.raises(ValueError, match="[Ww]avelength"):
        chrom_3d.extract_wavelengths(999)


def test_extract_wavelengths_rejects_chromatogram_with_no_3d_signal(chrom_2d):
    with pytest.raises(ValueError, match="3D|no 3D signal"):
        chrom_2d.extract_wavelengths(210)


def test_extract_wavelengths_finds_the_3d_signal_in_a_mixed_chromatogram(chrom_mixed):
    result = chrom_mixed.extract_wavelengths(210)
    np.testing.assert_array_equal(result.signals["PDA 210 nm"].signal, [1.0, 5.0, 9.0])


def test_extract_wavelengths_requires_a_detector_when_ambiguous():
    chrom = Chromatogram(
        signals={"PDA A": make_pda("PDA A"), "PDA B": make_pda("PDA B")},
        filename="test.lcd",
    )
    with pytest.raises(ValueError, match="[Mm]ultiple"):
        chrom.extract_wavelengths(210)


def test_extract_wavelengths_accepts_an_explicit_detector():
    chrom = Chromatogram(
        signals={"PDA A": make_pda("PDA A"), "PDA B": make_pda("PDA B")},
        filename="test.lcd",
    )
    result = chrom.extract_wavelengths(210, detector="PDA B")
    assert "PDA B 210 nm" in result.signals


def test_extract_wavelengths_rejects_unknown_detector(chrom_3d):
    with pytest.raises(ValueError, match="[Dd]etector"):
        chrom_3d.extract_wavelengths(210, detector="nope")


def test_extract_wavelengths_rejects_a_2d_detector(chrom_mixed):
    with pytest.raises(ValueError, match="3D|not a 3D"):
        chrom_mixed.extract_wavelengths(210, detector="RID")


def test_extracted_signal_is_displayable(chrom_3d):
    result = chrom_3d.extract_wavelengths(210)
    graphs = result.to_graph_properties()
    assert any(isinstance(g, GraphProperties2D) for g in graphs)
    assert any(isinstance(g, GraphProperties3D) for g in graphs)
