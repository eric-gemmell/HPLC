from __future__ import annotations

import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hplc.graphing.signal import plot_signal_2d, PALETTE


def plot_chromatogram(
    signals: dict[str, dict],
    title: str = "Chromatogram",
    cols: int = 2,
    subplot_width: int = 520,
    subplot_height: int = 340,
    shared_x: bool = False,
    filename: str | None = None,
    dpi: int = 300,
    style: str = "color",
) -> go.Figure:
    """
    Plot a tiled grid of chromatogram signals.

    Each subplot is rendered by plot_signal_2d.

    Parameters
    ----------
    signals : dict[str, dict]
        Mapping of detector name to signal data. Each value must contain:
            - time: array-like
            - time_unit: str
            - signal: array-like
            - signal_unit: str
    title : str
        Overall figure title.
    cols : int
        Number of columns in the grid.
    subplot_width, subplot_height : int
        Dimensions of each subplot in pixels.
    shared_x : bool
        Whether subplots share x-axes.
    filename : str | None
        If provided, save the figure as a PNG.
    dpi : int
        Resolution for PNG export.
    style : str
        "color" or "bw".
    """
    n = len(signals)
    if n == 0:
        raise ValueError("No signals to plot")

    cols = min(cols, n)
    rows = math.ceil(n / cols)

    subplot_titles = list(signals.keys())

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_x,
        vertical_spacing=0.12 if rows > 1 else 0.0,
        horizontal_spacing=0.08,
    )

    for i, (name, data) in enumerate(signals.items()):
        r = (i // cols) + 1
        c = (i % cols) + 1
        color = PALETTE[i % len(PALETTE)]

        signal_dict = {
            "name": name,
            "time": data["time"],
            "time_unit": data["time_unit"],
            "signal": data["signal"],
            "signal_unit": data["signal_unit"],
            "color": color,
        }

        plot_signal_2d(
            signals=[signal_dict],
            fig=fig,
            row=r,
            col=c,
            style=style,
        )

        # Configure axes for this subplot
        axis_suffix = "" if i == 0 else str(i + 1)
        fig.update_layout(**{
            f"xaxis{axis_suffix}": dict(
                title=dict(text=f"Time ({data['time_unit']})", font=dict(size=10)),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
                linecolor="#ccc",
                linewidth=1,
                tickfont=dict(size=9),
            ),
            f"yaxis{axis_suffix}": dict(
                title=dict(text=f"Signal ({data['signal_unit']})", font=dict(size=10)),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
                linecolor="#ccc",
                linewidth=1,
                tickfont=dict(size=9),
            ),
        })

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color="#2c3e50"),
            x=0.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=subplot_width * cols,
        height=subplot_height * rows,
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
