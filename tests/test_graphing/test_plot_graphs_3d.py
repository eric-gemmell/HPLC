from __future__ import annotations

import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import patch

from hplc.graphing import plot_graphs
from hplc.graphing.graph_models import (
    TraceProperties2D,
    TraceProperties3D,
    GraphProperties2D,
    GraphProperties3D,
)


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(go.Figure, "show", lambda self: None)


def _trace_2d(name="UV 254 nm") -> TraceProperties2D:
    time = np.linspace(0, 600, 50)
    return TraceProperties2D(
        name=name,
        time=time,
        time_unit="s",
        signal=np.sin(2 * np.pi * time / 200),
        signal_unit="mAU",
    )


def _trace_3d(name="PDA") -> TraceProperties3D:
    time = np.linspace(0, 600, 50)
    wavelengths = np.linspace(190, 600, 20)
    signal = np.outer(np.sin(2 * np.pi * time / 200), np.ones(20))
    return TraceProperties3D(
        name=name,
        time=time,
        time_unit="s",
        wavelengths=wavelengths,
        wavelength_unit="nm",
        signal=signal,
        signal_unit="mAU",
    )


# ---------------------------------------------------------------------------
# GraphProperties3D dataclass
# ---------------------------------------------------------------------------

class TestGraphProperties3D:

    def test_construction_defaults(self):
        trace = _trace_3d()
        gp = GraphProperties3D(signal=trace)
        assert gp.signal == trace
        assert gp.title == ""
        assert gp.x_min is None
        assert gp.x_max is None
        assert gp.y_min is None
        assert gp.y_max is None
        assert gp.z_min is None
        assert gp.z_max is None

    def test_construction_with_all_fields(self):
        gp = GraphProperties3D(
            signal=_trace_3d(), title="PDA", x_min=0, x_max=600,
            y_min=190, y_max=600, z_min=-1, z_max=1,
        )
        assert gp.title == "PDA"
        assert (gp.x_min, gp.x_max) == (0, 600)
        assert (gp.y_min, gp.y_max) == (190, 600)
        assert (gp.z_min, gp.z_max) == (-1, 1)


# ---------------------------------------------------------------------------
# plot_graphs dispatch
# ---------------------------------------------------------------------------

class TestPlotGraphsDispatch:

    def test_accepts_a_single_3d_graph(self):
        fig = plot_graphs(graphs=[GraphProperties3D(signal=_trace_3d())])
        assert isinstance(fig, go.Figure)

    def test_renders_a_heatmap_for_a_3d_graph(self):
        fig = plot_graphs(graphs=[GraphProperties3D(signal=_trace_3d())])
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_accepts_mixed_2d_and_3d_graphs(self):
        fig = plot_graphs(graphs=[
            GraphProperties2D(signals=[_trace_2d()], title="RID"),
            GraphProperties3D(signal=_trace_3d(), title="PDA"),
        ])
        assert isinstance(fig, go.Figure)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)
        assert any(isinstance(t, go.Scatter) for t in fig.data)

    def test_subplot_titles_come_from_both_graph_types(self):
        fig = plot_graphs(graphs=[
            GraphProperties2D(signals=[_trace_2d()], title="RID"),
            GraphProperties3D(signal=_trace_3d(), title="PDA"),
        ])
        titles = [a.text for a in fig.layout.annotations]
        assert "RID" in titles
        assert "PDA" in titles

    @patch("hplc.graphing.multi_plotting.plot_3d_graph")
    def test_3d_graph_delegates_to_plot_3d_graph(self, mock_3d):
        trace = _trace_3d()
        graph = GraphProperties3D(
            signal=trace, title="PDA", x_min=0, x_max=600,
            y_min=190, y_max=600, z_min=-1, z_max=1,
        )
        fig = plot_graphs(graphs=[graph])
        mock_3d.assert_called_once_with(
            signal=trace,
            fig=fig,
            row=1,
            col=1,
            x_min=0,
            x_max=600,
            y_min=190,
            y_max=600,
            z_min=-1,
            z_max=1,
        )

    @patch("hplc.graphing.multi_plotting.plot_2d_graph")
    def test_2d_graph_still_delegates_to_plot_2d_graph(self, mock_2d):
        trace = _trace_2d()
        graph = GraphProperties2D(signals=[trace], title="RID")
        fig = plot_graphs(graphs=[graph])
        mock_2d.assert_called_once_with(
            signals=[trace],
            fig=fig,
            row=1,
            col=1,
            style="color",
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
        )

    @patch("hplc.graphing.multi_plotting.plot_3d_graph")
    @patch("hplc.graphing.multi_plotting.plot_2d_graph")
    def test_mixed_graphs_are_placed_in_grid_order(self, mock_2d, mock_3d):
        plot_graphs(
            graphs=[
                GraphProperties2D(signals=[_trace_2d()], title="A"),
                GraphProperties3D(signal=_trace_3d(), title="B"),
                GraphProperties2D(signals=[_trace_2d()], title="C"),
            ],
            cols=2,
        )
        assert mock_2d.call_args_list[0].kwargs["row"] == 1
        assert mock_2d.call_args_list[0].kwargs["col"] == 1
        assert mock_3d.call_args_list[0].kwargs["row"] == 1
        assert mock_3d.call_args_list[0].kwargs["col"] == 2
        assert mock_2d.call_args_list[1].kwargs["row"] == 2
        assert mock_2d.call_args_list[1].kwargs["col"] == 1

    def test_rejects_empty_graph_list(self):
        with pytest.raises(ValueError):
            plot_graphs(graphs=[])

    def test_rejects_unknown_graph_type(self):
        with pytest.raises(TypeError):
            plot_graphs(graphs=["not a graph"])
