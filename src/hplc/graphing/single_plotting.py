from __future__ import annotations

import plotly.graph_objects as go
import numpy as np

from hplc.graphing.graph_models import TraceProperties2D


PALETTE = [
    "#4C78A8", "#F58518", "#E45756", "#72B7B2",
    "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6",
]

BW_LINES = [
    "black",
    "dimgray",
    "gray",
    "darkgray",
]

FILL_PALETTE = [
    "rgba(228, 87, 86, 0.25)",
    "rgba(114, 183, 178, 0.25)",
    "rgba(245, 133, 24, 0.25)",
    "rgba(178, 121, 162, 0.25)",
    "rgba(84, 162, 75, 0.25)",
    "rgba(238, 202, 59, 0.25)",
]

FILL_BORDER_PALETTE = [
    "rgba(228, 87, 86, 0.6)",
    "rgba(114, 183, 178, 0.6)",
    "rgba(245, 133, 24, 0.6)",
    "rgba(178, 121, 162, 0.6)",
    "rgba(84, 162, 75, 0.6)",
    "rgba(238, 202, 59, 0.6)",
]

BW_FILL_PALETTE = [
    "rgba(0, 0, 0, 0.10)",
    "rgba(80, 80, 80, 0.10)",
    "rgba(160, 160, 160, 0.10)",
]

BW_FILL_BORDER_PALETTE = [
    "rgba(0, 0, 0, 0.5)",
    "rgba(80, 80, 80, 0.5)",
    "rgba(160, 160, 160, 0.5)",
]


def _resolve_style(style: str, n_signals: int):
    if style == "bw":
        lines = [BW_LINES[i % len(BW_LINES)] for i in range(n_signals)]
        fills = BW_FILL_PALETTE
        borders = BW_FILL_BORDER_PALETTE
    else:
        lines = [PALETTE[i % len(PALETTE)] for i in range(n_signals)]
        fills = FILL_PALETTE
        borders = FILL_BORDER_PALETTE
    return lines, fills, borders


def plot_2d_graph(
    signals: list[TraceProperties2D],
    title: str = "",
    width: int = 520,
    height: int = 340,
    filename: str | None = None,
    dpi: int = 300,
    style: str = "color",
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Core 2D signal renderer. Draws pre-computed data only."""
    if not signals:
        raise ValueError("No signals to plot")

    standalone = fig is None
    if standalone:
        fig = go.Figure()

    n = len(signals)
    line_colors, fill_palette, border_palette = _resolve_style(style, n)

    trace_kwargs = {}
    if row is not None and col is not None:
        trace_kwargs = {"row": row, "col": col}

    for i, s in enumerate(signals):
        time = np.asarray(s.time, dtype=float)
        signal = np.asarray(s.signal, dtype=float)
        color = s.color or line_colors[i]

        # Main signal line
        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal,
                mode="lines",
                line=dict(color=color, width=1.5),
                name=s.name,
                legendgroup=s.name,
                hovertemplate=(
                    f"<b>{s.name}</b><br>"
                    f"Time: %{{x:.3f}} {s.time_unit}<br>"
                    f"Signal: %{{y:.2f}} {s.signal_unit}"
                    "<extra></extra>"
                ),
            ),
            **trace_kwargs,
        )

        # Baseline
        if s.baseline is not None:
            baseline = np.asarray(s.baseline, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=baseline,
                    mode="lines",
                    line=dict(color=color, width=1, dash="dash"),
                    name=f"{s.name} baseline",
                    legendgroup=s.name,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                **trace_kwargs,
            )

        # Fill regions
        for j, fill in enumerate(s.fills or []):
            fill_time = np.asarray(fill.time, dtype=float)
            upper = np.asarray(fill.upper, dtype=float)
            lower = np.asarray(fill.lower, dtype=float) if fill.lower is not None else np.zeros_like(upper)

            fill_color = fill_palette[j % len(fill_palette)]
            border_color = border_palette[j % len(border_palette)]

            fig.add_trace(
                go.Scatter(
                    x=fill_time,
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    legendgroup=s.name,
                    hoverinfo="skip",
                ),
                **trace_kwargs,
            )
            fig.add_trace(
                go.Scatter(
                    x=fill_time,
                    y=upper,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=fill_color,
                    line=dict(color=border_color, width=1),
                    name=fill.label or f"Fill {j + 1}",
                    legendgroup=s.name,
                    showlegend=bool(fill.label),
                    hoverinfo="skip",
                ),
                **trace_kwargs,
            )

    if standalone:
        display_title = title or (signals[0].name if n == 1 else "")
        xaxis_opts = dict(
            title=dict(text=f"Time ({signals[0].time_unit})", font=dict(size=11)),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            linecolor="#ccc",
            linewidth=1,
        )
        yaxis_opts = dict(
            title=dict(text=f"Signal ({signals[0].signal_unit})", font=dict(size=11)),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            linecolor="#ccc",
            linewidth=1,
        )

        all_time = np.concatenate([np.asarray(s.time) for s in signals])
        all_signal = np.concatenate([np.asarray(s.signal) for s in signals])

        if x_min is not None or x_max is not None:
            xaxis_opts["range"] = [
                x_min if x_min is not None else float(all_time.min()),
                x_max if x_max is not None else float(all_time.max()),
            ]

        if y_min is not None or y_max is not None:
            yaxis_opts["range"] = [
                y_min if y_min is not None else float(all_signal.min()),
                y_max if y_max is not None else float(all_signal.max()),
            ]

        fig.update_layout(
            title=dict(
                text=display_title,
                font=dict(size=14, color="#2c3e50"),
                x=0.5,
            ),
            xaxis=xaxis_opts,
            yaxis=yaxis_opts,
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=width,
            height=height,
            margin=dict(l=55, r=20, t=40, b=45),
            showlegend=n > 1 or any(s.fills for s in signals),
            hovermode="x unified",
        )

        if filename:
            scale = dpi / 96
            fig.write_image(filename, scale=scale)

        fig.show()

    return fig
