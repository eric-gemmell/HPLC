from __future__ import annotations

import pytest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unittest.mock import patch

from hplc.graphing import plot_2d_graph
from hplc.graphing.graph_models import TraceProperties2D, FillRegion


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(go.Figure, "show", lambda self: None)


def _trace(name="UV 254 nm") -> TraceProperties2D:
    time = np.linspace(0, 600, 300)
    return TraceProperties2D(
        name=name,
        time=time,
        time_unit="s",
        signal=np.sin(2 * np.pi * time / 200),
        signal_unit="mAU",
    )


def _trace_with_fills(name="UV 254 nm") -> TraceProperties2D:
    time = np.linspace(0, 600, 300)
    return TraceProperties2D(
        name=name,
        time=time,
        time_unit="s",
        signal=np.sin(2 * np.pi * time / 200),
        signal_unit="mAU",
        fills=[FillRegion(
            time=time[50:100],
            upper=np.abs(np.sin(2 * np.pi * time[50:100] / 200)),
            label="Peak @ 100s",
        )],
        baseline=np.zeros_like(time),
    )


class TestAcceptsParameters:

    def test_single_signal(self):
        fig = plot_2d_graph(signals=[_trace()])
        assert isinstance(fig, go.Figure)

    def test_multiple_signals(self):
        fig = plot_2d_graph(signals=[_trace("A"), _trace("B")])
        assert isinstance(fig, go.Figure)

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError):
            plot_2d_graph(signals=[])

    def test_title(self):
        fig = plot_2d_graph(signals=[_trace()], title="My Title")
        assert isinstance(fig, go.Figure)

    def test_width_height_dpi(self):
        fig = plot_2d_graph(signals=[_trace()], width=800, height=600, dpi=150)
        assert isinstance(fig, go.Figure)

    def test_style(self):
        fig = plot_2d_graph(signals=[_trace()], style="bw")
        assert isinstance(fig, go.Figure)

    def test_all_parameters(self):
        traces = [_trace_with_fills("DAD 190 nm"), _trace_with_fills("DAD 254 nm"), _trace_with_fills("DAD 405 nm")]
        fig = plot_2d_graph(
            signals=traces,
            title="HEY bby",
            width=800,
            height=500,
            filename=None,
            dpi=300,
            style="bw",
            x_min=300,
            x_max=None,
            y_min=0,
            y_max=None,
        )
        assert isinstance(fig, go.Figure)
        assert all(isinstance(t, TraceProperties2D) for t in traces)
        assert all(isinstance(f, FillRegion) for t in traces for f in t.fills)
        assert all(isinstance(t.baseline, np.ndarray) for t in traces)

    def test_x_min_and_x_max(self):
        fig = plot_2d_graph(signals=[_trace()], x_min=100, x_max=400)
        assert isinstance(fig, go.Figure)

    def test_partial_x_limits(self):
        fig = plot_2d_graph(signals=[_trace()], x_min=100)
        assert isinstance(fig, go.Figure)
    

    def test_partial_y_limits(self):
        fig = plot_2d_graph(signals=[_trace()], y_max=0.5)
        assert isinstance(fig, go.Figure)

    def test_fills_and_baseline(self):
        fig = plot_2d_graph(signals=[_trace_with_fills()])
        assert isinstance(fig, go.Figure)

    def test_color_override(self):
        t = _trace()
        t.color = "#FF0000"
        fig = plot_2d_graph(signals=[t])
        assert isinstance(fig, go.Figure)


class TestFigureInjection:

    def test_creates_figure_when_none(self):
        fig = plot_2d_graph(signals=[_trace()])
        assert isinstance(fig, go.Figure)

    def test_draws_onto_existing_figure(self):
        existing = make_subplots(rows=1, cols=2)
        fig = plot_2d_graph(signals=[_trace()], fig=existing, row=1, col=2)
        assert fig is existing
        assert len(fig.data) > 0

    def test_does_not_show_when_fig_provided(self):
        existing = make_subplots(rows=1, cols=1)
        with patch.object(go.Figure, "show") as mock_show:
            plot_2d_graph(signals=[_trace()], fig=existing, row=1, col=1)
            mock_show.assert_not_called()

    def test_shows_when_standalone(self):
        with patch.object(go.Figure, "show") as mock_show:
            plot_2d_graph(signals=[_trace()])
            mock_show.assert_called_once()
