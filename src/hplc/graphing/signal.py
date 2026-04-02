from __future__ import annotations

import plotly.graph_objects as go
import numpy as np
from scipy.special import erf


PALETTE = [
    "#4C78A8", "#F58518", "#E45756", "#72B7B2",
    "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6",
]

PEAK_FILLS = [
    "rgba(228, 87, 86, 0.25)",
    "rgba(114, 183, 178, 0.25)",
    "rgba(245, 133, 24, 0.25)",
    "rgba(178, 121, 162, 0.25)",
    "rgba(84, 162, 75, 0.25)",
    "rgba(238, 202, 59, 0.25)",
]

PEAK_BORDERS = [
    "rgba(228, 87, 86, 0.6)",
    "rgba(114, 183, 178, 0.6)",
    "rgba(245, 133, 24, 0.6)",
    "rgba(178, 121, 162, 0.6)",
    "rgba(84, 162, 75, 0.6)",
    "rgba(238, 202, 59, 0.6)",
]

BW_LINES = [
    "black",
    "dimgray",
    "gray",
    "darkgray",
]

BW_PEAK_FILLS = [
    "rgba(0, 0, 0, 0.10)",
    "rgba(80, 80, 80, 0.10)",
    "rgba(160, 160, 160, 0.10)",
]

BW_PEAK_BORDERS = [
    "rgba(0, 0, 0, 0.5)",
    "rgba(80, 80, 80, 0.5)",
    "rgba(160, 160, 160, 0.5)",
]


def _skew_normal_curve(time, mu, amplitude, sigma, alpha):
    z = (time - mu) / sigma
    phi = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z ** 2)
    Phi = 0.5 * (1 + erf(alpha * z / np.sqrt(2)))
    return 2 * amplitude * phi * Phi


def _trim_to_area_fraction(time, curve, fraction=0.99):
    total_area = np.trapezoid(curve, time)
    if total_area <= 0:
        return np.ones(len(time), dtype=bool)

    cumulative = np.cumsum(0.5 * (curve[:-1] + curve[1:]) * np.diff(time))
    cumulative = np.insert(cumulative, 0, 0)

    lower = (1 - fraction) / 2 * total_area
    upper = (1 + fraction) / 2 * total_area

    return (cumulative >= lower) & (cumulative <= upper)


def _resolve_style(style: str, n_signals: int):
    """Return (line_colors, peak_fills, peak_borders) lists for n signals."""
    if style == "bw":
        lines = [BW_LINES[i % len(BW_LINES)] for i in range(n_signals)]
        fills = BW_PEAK_FILLS
        borders = BW_PEAK_BORDERS
    else:
        lines = [PALETTE[i % len(PALETTE)] for i in range(n_signals)]
        fills = PEAK_FILLS
        borders = PEAK_BORDERS
    return lines, fills, borders


def plot_signal_2d(
    signals: list[dict],
    title: str = "",
    width: int = 520,
    height: int = 340,
    filename: str | None = None,
    dpi: int = 300,
    style: str = "color",
    x_lim: tuple[float | None, float | None] | None = None,
    y_lim: tuple[float | None, float | None] | None = None,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """
    Core 2D signal renderer.

    Parameters
    ----------
    signals : list[dict]
        Each dict must contain:
            - name: str
            - time: array-like
            - time_unit: str
            - signal: array-like
            - signal_unit: str
        Optional per-signal keys:
            - peaks: list[dict] — {mu, amplitude, sigma, alpha, label?}
            - delta: float (time shift, default 0)
            - stretch_factor: float (time scale, default 1)
            - y_offset: float (vertical offset, default 0)
            - scale: float (signal amplitude multiplier, default 1)
            - color: str | None (override line color)
    title : str
        Figure title (only applied for standalone figures).
    width, height : int
        Figure dimensions in pixels.
    filename : str | None
        If provided, export as PNG.
    dpi : int
        Resolution for PNG export.
    style : str
        "color" or "bw".
    fig : go.Figure | None
        If provided, draw onto this figure instead of creating a new one.
    row, col : int | None
        Subplot position when drawing onto an existing figure.
    """
    if not signals:
        raise ValueError("No signals to plot")

    standalone = fig is None
    if standalone:
        fig = go.Figure()

    n = len(signals)
    line_colors, peak_fill_palette, peak_border_palette = _resolve_style(style, n)

    trace_kwargs = {}
    if row is not None and col is not None:
        trace_kwargs = {"row": row, "col": col}

    for i, s in enumerate(signals):
        raw_time = np.asarray(s["time"], dtype=float)
        raw_signal = np.asarray(s["signal"], dtype=float)
        time_unit = s["time_unit"]
        signal_unit = s["signal_unit"]
        name = s["name"]
        peaks = s.get("peaks") or []
        delta = s.get("delta", 0.0)
        stretch_factor = s.get("stretch_factor", 1.0)
        y_offset = s.get("y_offset", 0.0)
        scale = s.get("scale", 1.0)
        color = s.get("color") or line_colors[i]

        time = stretch_factor * raw_time + delta
        signal = raw_signal * scale + y_offset

        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal,
                mode="lines",
                line=dict(color=color, width=1.5),
                name=name,
                legendgroup=name,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Time: %{{x:.3f}} {time_unit}<br>"
                    f"Signal: %{{customdata:.2f}} {signal_unit}"
                    "<extra></extra>"
                ),
                customdata=raw_signal * scale,
            ),
            **trace_kwargs,
        )

        for j, p in enumerate(peaks):
            curve = _skew_normal_curve(raw_time, p["mu"], p["amplitude"], p["sigma"], p["alpha"])
            mask = _trim_to_area_fraction(raw_time, curve, fraction=0.99)

            t_trimmed = stretch_factor * raw_time[mask] + delta
            c_trimmed = curve[mask] * scale + y_offset
            baseline = np.full_like(c_trimmed, y_offset)

            fill_color = peak_fill_palette[j % len(peak_fill_palette)]
            border_color = peak_border_palette[j % len(peak_border_palette)]
            label = p.get("label", f"Peak {j + 1}")

            fig.add_trace(
                go.Scatter(
                    x=t_trimmed,
                    y=baseline,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    legendgroup=name,
                    hoverinfo="skip",
                ),
                **trace_kwargs,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_trimmed,
                    y=c_trimmed,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=fill_color,
                    line=dict(color=border_color, width=1),
                    name=label,
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"Time: %{{x:.3f}} {time_unit}<br>"
                        f"Fit: %{{customdata:.2f}} {signal_unit}"
                        "<extra></extra>"
                    ),
                    customdata=curve[mask] * scale,
                ),
                **trace_kwargs,
            )

    if standalone:
        display_title = title or (signals[0]["name"] if n == 1 else "")
        xaxis_opts = dict(
            title=dict(text=f"Time ({signals[0]['time_unit']})", font=dict(size=11)),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            linecolor="#ccc",
            linewidth=1,
        )
        yaxis_opts = dict(
            title=dict(text=f"Signal ({signals[0]['signal_unit']})", font=dict(size=11)),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            linecolor="#ccc",
            linewidth=1,
        )

        if x_lim is not None:
            lo, hi = x_lim
            if lo is not None or hi is not None:
                xaxis_opts["range"] = [lo, hi]

        if y_lim is not None:
            lo, hi = y_lim
            if lo is not None or hi is not None:
                yaxis_opts["range"] = [lo, hi]

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
            showlegend=n > 1 or any(s.get("peaks") for s in signals),
            hovermode="x unified",
        )

        if filename:
            scale = dpi / 96
            fig.write_image(filename, scale=scale)

        fig.show()

    return fig
