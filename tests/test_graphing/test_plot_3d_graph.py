from __future__ import annotations

import pytest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unittest.mock import patch

from hplc.graphing import plot_3d_graph
from hplc.graphing.graph_models import TraceProperties3D


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(go.Figure, "show", lambda self: None)


def _trace(name="PDA 190-600 nm") -> TraceProperties3D:
    time = np.linspace(0, 600, 60)
    wavelengths = np.linspace(190, 600, 40)
    signal = np.sin(2 * np.pi * time[:, None] / 200) * np.exp(-((wavelengths[None, :] - 254) ** 2) / 2000)
    return TraceProperties3D(
        name=name,
        time=time,
        time_unit="s",
        wavelengths=wavelengths,
        wavelength_unit="nm",
        signal=signal,
        signal_unit="mAU",
    )


def _trace_with_colors(name="PDA 190-600 nm") -> TraceProperties3D:
    trace = _trace(name)
    trace.colors = ["#FFFFFF", "#4C78A8", "#000000"]
    trace.color_values = np.array([-1.0, 0.0, 1.0])
    return trace


class TestAcceptsParameters:

    def test_single_signal(self):
        fig = plot_3d_graph(signal=_trace())
        assert isinstance(fig, go.Figure)

    def test_rejects_none(self):
        with pytest.raises(ValueError):
            plot_3d_graph(signal=None)

    def test_title(self):
        fig = plot_3d_graph(signal=_trace(), title="My Title")
        assert isinstance(fig, go.Figure)

    def test_width_height_dpi(self):
        fig = plot_3d_graph(signal=_trace(), width=800, height=600, dpi=150)
        assert isinstance(fig, go.Figure)

    def test_filename_defaults_to_none(self):
        fig = plot_3d_graph(signal=_trace(), filename=None)
        assert isinstance(fig, go.Figure)

    def test_x_min_and_x_max(self):
        fig = plot_3d_graph(signal=_trace(), x_min=100, x_max=400)
        assert isinstance(fig, go.Figure)

    def test_y_min_and_y_max(self):
        fig = plot_3d_graph(signal=_trace(), y_min=200, y_max=400)
        assert isinstance(fig, go.Figure)

    def test_z_min_and_z_max(self):
        fig = plot_3d_graph(signal=_trace(), z_min=-0.5, z_max=0.5)
        assert isinstance(fig, go.Figure)

    def test_partial_x_limits(self):
        fig = plot_3d_graph(signal=_trace(), x_min=100)
        assert isinstance(fig, go.Figure)

    def test_partial_y_limits(self):
        fig = plot_3d_graph(signal=_trace(), y_max=400)
        assert isinstance(fig, go.Figure)

    def test_partial_z_limits(self):
        fig = plot_3d_graph(signal=_trace(), z_max=0.5)
        assert isinstance(fig, go.Figure)

    def test_colors_from_trace_properties(self):
        fig = plot_3d_graph(signal=_trace_with_colors())
        assert isinstance(fig, go.Figure)

    def test_all_parameters(self):
        trace = _trace_with_colors("PDA 190-600 nm")
        fig = plot_3d_graph(
            signal=trace,
            title="HEY bby",
            width=800,
            height=500,
            filename=None,
            dpi=300,
            x_min=100,
            x_max=None,
            y_min=200,
            y_max=None,
            z_min=-0.5,
            z_max=None,
        )
        assert isinstance(fig, go.Figure)
        assert isinstance(trace, TraceProperties3D)
        assert isinstance(trace.signal, np.ndarray)
        assert trace.signal.ndim == 2


class TestFigureInjection:

    def test_creates_figure_when_none(self):
        fig = plot_3d_graph(signal=_trace())
        assert isinstance(fig, go.Figure)

    def test_draws_onto_existing_figure(self):
        existing = make_subplots(rows=1, cols=2)
        fig = plot_3d_graph(signal=_trace(), fig=existing, row=1, col=2)
        assert fig is existing
        assert len(fig.data) > 0

    def test_does_not_show_when_fig_provided(self):
        existing = make_subplots(rows=1, cols=1)
        with patch.object(go.Figure, "show") as mock_show:
            plot_3d_graph(signal=_trace(), fig=existing, row=1, col=1)
            mock_show.assert_not_called()

    def test_shows_when_standalone(self):
        with patch.object(go.Figure, "show") as mock_show:
            plot_3d_graph(signal=_trace())
            mock_show.assert_called_once()


class TestFigureProperties:
    """A few assertions on the returned figure, mirroring the 2D line-width tests."""

    def test_default_width_and_height(self):
        fig = plot_3d_graph(signal=_trace())
        assert fig.layout.width == 520
        assert fig.layout.height == 340

    def test_width_and_height_override(self):
        fig = plot_3d_graph(signal=_trace(), width=800, height=600)
        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_z_limits_clip_the_colour_scale(self):
        fig = plot_3d_graph(signal=_trace(), z_min=-0.5, z_max=0.5)
        assert fig.data[0].zmin == -0.5
        assert fig.data[0].zmax == 0.5
