from __future__ import annotations

import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import patch, call

from hplc.graphing import plot_graphs
from hplc.graphing.graph_models import TraceProperties2D, GraphProperties2D


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


def _graph(title="Graph A", n_traces=1) -> GraphProperties2D:
    traces = [_trace(f"Trace {i}") for i in range(n_traces)]
    return GraphProperties2D(signals=traces, title=title)


# ---------------------------------------------------------------------------
# GraphProperties2D dataclass
# ---------------------------------------------------------------------------

class TestGraphProperties2D:

    def test_construction_defaults(self):
        traces = [_trace()]
        gp = GraphProperties2D(signals=traces)
        assert gp.signals == traces
        assert gp.title == ""
        assert gp.x_min is None
        assert gp.x_max is None
        assert gp.y_min is None
        assert gp.y_max is None

    def test_construction_with_all_fields(self):
        traces = [_trace("A"), _trace("B")]
        gp = GraphProperties2D(
            signals=traces,
            title="My Graph",
            x_min=10.0,
            x_max=500.0,
            y_min=-1.0,
            y_max=1.0,
        )
        assert gp.title == "My Graph"
        assert gp.x_min == 10.0
        assert gp.x_max == 500.0
        assert gp.y_min == -1.0
        assert gp.y_max == 1.0


# ---------------------------------------------------------------------------
# plot_graphs — accepts parameters
# ---------------------------------------------------------------------------

class TestAcceptsParameters:

    def test_single_graph(self):
        fig = plot_graphs(graphs=[_graph()])
        assert isinstance(fig, go.Figure)

    def test_multiple_graphs(self):
        fig = plot_graphs(graphs=[_graph("A"), _graph("B"), _graph("C")])
        assert isinstance(fig, go.Figure)

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError):
            plot_graphs(graphs=[])

    def test_title(self):
        fig = plot_graphs(graphs=[_graph()], title="Overall Title")
        assert isinstance(fig, go.Figure)

    def test_cols(self):
        fig = plot_graphs(graphs=[_graph("A"), _graph("B"), _graph("C")], cols=3)
        assert isinstance(fig, go.Figure)

    def test_cols_clamped_to_graph_count(self):
        fig = plot_graphs(graphs=[_graph("A")], cols=5)
        assert isinstance(fig, go.Figure)

    def test_plot_dimensions(self):
        fig = plot_graphs(graphs=[_graph()], width=800, height=600)
        assert isinstance(fig, go.Figure)

    def test_style(self):
        fig = plot_graphs(graphs=[_graph()], style="bw")
        assert isinstance(fig, go.Figure)

    def test_dpi(self):
        fig = plot_graphs(graphs=[_graph()], dpi=150)
        assert isinstance(fig, go.Figure)

    def test_shared_x(self):
        fig = plot_graphs(graphs=[_graph("A"), _graph("B")], shared_x=True)
        assert isinstance(fig, go.Figure)

    def test_all_parameters(self):
        graphs = [
            GraphProperties2D(signals=[_trace("UV")], title="UV", x_min=100, x_max=400),
            GraphProperties2D(signals=[_trace("Vis")], title="Vis", y_min=0, y_max=10),
        ]
        fig = plot_graphs(
            graphs=graphs,
            cols=2,
            title="Full chromatogram",
            width=600,
            height=400,
            filename=None,
            dpi=300,
            style="bw",
            shared_x=True,
        )
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_graphs — delegates to plot_2d_graph
# ---------------------------------------------------------------------------

class TestDelegation:

    @patch("hplc.graphing.multi_plotting.plot_2d_graph")
    def test_calls_plot_2d_graph_once_per_graph(self, mock_plot):
        graphs = [_graph("A"), _graph("B"), _graph("C")]
        plot_graphs(graphs=graphs)
        assert mock_plot.call_count == len(graphs)

    @patch("hplc.graphing.multi_plotting.plot_2d_graph")
    def test_passes_graph_signals_and_position(self, mock_plot):
        g1 = _graph("A")
        g2 = _graph("B")
        plot_graphs(graphs=[g1, g2], cols=2)

        calls = mock_plot.call_args_list
        # First graph at row=1, col=1
        assert calls[0].kwargs["signals"] == g1.signals
        assert calls[0].kwargs["row"] == 1
        assert calls[0].kwargs["col"] == 1
        # Second graph at row=1, col=2
        assert calls[1].kwargs["signals"] == g2.signals
        assert calls[1].kwargs["row"] == 1
        assert calls[1].kwargs["col"] == 2

    @patch("hplc.graphing.multi_plotting.plot_2d_graph")
    def test_passes_per_graph_axis_limits(self, mock_plot):
        g = GraphProperties2D(
            signals=[_trace()],
            title="UV",
            x_min=100,
            x_max=400,
            y_min=-1,
            y_max=5,
        )
        plot_graphs(graphs=[g])

        kwargs = mock_plot.call_args.kwargs
        assert kwargs["x_min"] == 100
        assert kwargs["x_max"] == 400
        assert kwargs["y_min"] == -1
        assert kwargs["y_max"] == 5

    @patch("hplc.graphing.multi_plotting.plot_2d_graph")
    def test_passes_style_to_each_subplot(self, mock_plot):
        plot_graphs(graphs=[_graph("A"), _graph("B")], style="bw")
        for c in mock_plot.call_args_list:
            assert c.kwargs["style"] == "bw"

    @patch("hplc.graphing.multi_plotting.plot_2d_graph")
    def test_passes_same_fig_to_all(self, mock_plot):
        plot_graphs(graphs=[_graph("A"), _graph("B")])
        figs = [c.kwargs["fig"] for c in mock_plot.call_args_list]
        assert figs[0] is figs[1]


# ---------------------------------------------------------------------------
# plot_graphs — layout
# ---------------------------------------------------------------------------

class TestLayout:

    def test_grid_layout_2_cols(self):
        graphs = [_graph(f"G{i}") for i in range(3)]
        fig = plot_graphs(graphs=graphs, cols=2)
        # 3 graphs in 2 cols = 2 rows
        assert isinstance(fig, go.Figure)

    def test_single_graph_no_subplot_overhead(self):
        fig = plot_graphs(graphs=[_graph()])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_subplot_titles_from_graph_properties(self):
        graphs = [_graph("UV 254"), _graph("UV 280")]
        fig = plot_graphs(graphs=graphs, cols=2)
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert "UV 254" in annotation_texts
        assert "UV 280" in annotation_texts

    def test_shows_figure(self):
        with patch.object(go.Figure, "show") as mock_show:
            plot_graphs(graphs=[_graph()])
            mock_show.assert_called_once()
