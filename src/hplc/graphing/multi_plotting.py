from __future__ import annotations

import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hplc.graphing.single_plotting import plot_2d_graph, plot_3d_graph
from hplc.graphing.graph_models import GraphProperties2D, GraphProperties3D


def plot_graphs(
    graphs: list[GraphProperties2D | GraphProperties3D],
    cols: int = 2,
    title: str = "",
    width: int = 520,
    height: int = 340,
    shared_x: bool = False,
    filename: str | None = None,
    dpi: int = 300,
    style: str = "color",
) -> go.Figure:
    n = len(graphs)
    if n == 0:
        raise ValueError("No graphs to plot")

    for g in graphs:
        if not isinstance(g, (GraphProperties2D, GraphProperties3D)):
            raise TypeError(
                f"graphs must contain GraphProperties2D or GraphProperties3D, "
                f"got {type(g).__name__}"
            )

    cols = min(cols, n)
    rows = math.ceil(n / cols)

    subplot_titles = [g.title for g in graphs]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_x,
        vertical_spacing=0.12 if rows > 1 else 0.0,
        horizontal_spacing=0.08,
    )

    for i, g in enumerate(graphs):
        r = (i // cols) + 1
        c = (i % cols) + 1

        if isinstance(g, GraphProperties2D):
            plot_2d_graph(
                signals=g.signals,
                fig=fig,
                row=r,
                col=c,
                style=style,
                x_min=g.x_min,
                x_max=g.x_max,
                y_min=g.y_min,
                y_max=g.y_max,
            )
        elif isinstance(g, GraphProperties3D):
            plot_3d_graph(
                signal=g.signal,
                fig=fig,
                row=r,
                col=c,
                x_min=g.x_min,
                x_max=g.x_max,
                y_min=g.y_min,
                y_max=g.y_max,
                z_min=g.z_min,
                z_max=g.z_max,
            )
        else:
            raise TypeError(
                f"graphs must contain GraphProperties2D or GraphProperties3D, "
                f"got {type(g).__name__}"
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color="#2c3e50"),
            x=0.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width * cols,
        height=height * rows,
        margin=dict(l=55, r=20, t=60, b=45),
        hovermode="x unified",
        showlegend=False,
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=11, color="#2c3e50")

    if filename:
        scale = dpi / 96
        fig.write_image(filename, scale=scale)

    fig.show()
    return fig
