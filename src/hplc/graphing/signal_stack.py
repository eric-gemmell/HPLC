from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from HPLC.graphing.signal import _skew_normal_curve, _trim_to_area_fraction


TRACE_PALETTE = [
    "#4C78A8", "#F58518", "#E45756", "#72B7B2",
    "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6",
]

PEAK_FILLS = [
    "rgba(228, 87, 86, 0.20)",
    "rgba(114, 183, 178, 0.20)",
    "rgba(245, 133, 24, 0.20)",
    "rgba(178, 121, 162, 0.20)",
    "rgba(84, 162, 75, 0.20)",
    "rgba(238, 202, 59, 0.20)",
]

PEAK_BORDERS = [
    "rgba(228, 87, 86, 0.50)",
    "rgba(114, 183, 178, 0.50)",
    "rgba(245, 133, 24, 0.50)",
    "rgba(178, 121, 162, 0.50)",
    "rgba(84, 162, 75, 0.50)",
    "rgba(238, 202, 59, 0.50)",
]


def plot_signal_stack_2d(
    signals: list[dict],
    title: str = "Signal Stack",
    scrunch: float = 1.0,
    width: int = 700,
    height: int | None = None,
    filename: str | None = None,
    dpi: int = 300,
) -> go.Figure:
    """
    Plot multiple 2D signals as a vertically stacked overlay.

    Each signal is offset vertically so traces don't overlap (at scrunch=1.0).
    Increasing scrunch compresses the vertical spacing, bringing traces closer
    together. Peaks, if present, are drawn at the corresponding offset.

    Per-signal time alignment is applied via optional ``delta`` (time shift)
    and ``stretch_factor`` (time scaling) keys. The displayed time for each
    point becomes ``stretch_factor * original_time + delta``.

    Parameters
    ----------
    signals : list[dict]
        Each dict must contain:
            - name: str
            - time: array-like
            - time_unit: str
            - signal: array-like
            - signal_unit: str
            - peaks: list[dict] | None  — {mu, amplitude, sigma, alpha, label?}
            - delta: float (optional, default 0.0)
            - stretch_factor: float (optional, default 1.0)
    title : str
        Figure title.
    scrunch : float
        Controls vertical compression. 1.0 = no overlap, >1 = tighter,
        <1 = more spread out.
    width : int
        Figure width in pixels.
    height : int | None
        Figure height. Defaults to 200 + 120 per signal.
    filename : str | None
        If provided, export as PNG at this path.
    dpi : int
        Resolution for PNG export.
    """
    n = len(signals)
    if n == 0:
        raise ValueError("No signals to plot")

    if height is None:
        height = 200 + 120 * n

    # Determine vertical spacing between traces.
    # At scrunch=1.0, spacing equals the max amplitude across all signals.
    max_amplitude = max(
        np.ptp(np.asarray(s["signal"], dtype=float)) for s in signals
    )
    if max_amplitude == 0:
        max_amplitude = 1.0

    spacing = max_amplitude / scrunch

    fig = go.Figure()

    y_tick_vals = []
    y_tick_labels = []

    for i, s in enumerate(signals):
        raw_time = np.asarray(s["time"], dtype=float)
        signal = np.asarray(s["signal"], dtype=float)
        time_unit = s["time_unit"]
        signal_unit = s["signal_unit"]
        name = s["name"]
        peaks = s.get("peaks") or []
        delta = s.get("delta", 0.0)
        stretch_factor = s.get("stretch_factor", 1.0)

        # Apply time alignment
        time = stretch_factor * raw_time + delta

        y_offset = i * spacing
        y_shifted = signal + y_offset

        color = TRACE_PALETTE[i % len(TRACE_PALETTE)]

        fig.add_trace(
            go.Scatter(
                x=time,
                y=y_shifted,
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
                customdata=signal,
            )
        )

        # Draw peaks at the same vertical offset, with aligned time
        for j, p in enumerate(peaks):
            curve = _skew_normal_curve(raw_time, p["mu"], p["amplitude"], p["sigma"], p["alpha"])
            mask = _trim_to_area_fraction(raw_time, curve, fraction=0.99)

            t_trimmed = stretch_factor * raw_time[mask] + delta
            c_trimmed = curve[mask] + y_offset

            baseline = np.full_like(c_trimmed, y_offset)

            fill_color = PEAK_FILLS[j % len(PEAK_FILLS)]
            border_color = PEAK_BORDERS[j % len(PEAK_BORDERS)]
            label = p.get("label", f"Peak {j + 1}")

            # Invisible baseline, then filled curve on top
            fig.add_trace(
                go.Scatter(
                    x=t_trimmed,
                    y=baseline,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    legendgroup=name,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=t_trimmed,
                    y=c_trimmed,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=fill_color,
                    line=dict(color=border_color, width=0.8),
                    name=f"{name} — {label}",
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{name} — {label}</b><br>"
                        f"Time: %{{x:.3f}} {time_unit}<br>"
                        f"Fit: %{{customdata:.2f}} {signal_unit}"
                        "<extra></extra>"
                    ),
                    customdata=curve[mask],
                )
            )

        y_tick_vals.append(y_offset)
        y_tick_labels.append(name)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color="#2c3e50"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text=f"Time ({signals[0]['time_unit']})", font=dict(size=11)),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            linecolor="#ccc",
            linewidth=1,
        ),
        yaxis=dict(
            tickvals=y_tick_vals,
            ticktext=y_tick_labels,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            linecolor="#ccc",
            linewidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
        margin=dict(l=120, r=20, t=50, b=45),
        hovermode="x unified",
        showlegend=True,
        legend=dict(font=dict(size=9)),
    )

    if filename:
        scale = dpi / 96
        fig.write_image(filename, scale=scale)

    fig.show()
    return fig
