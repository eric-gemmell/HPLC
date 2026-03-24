import plotly.graph_objects as go
import numpy as np
from scipy.special import erf


def _skew_normal_curve(time, mu, amplitude, sigma, alpha):
    z = (time - mu) / sigma
    phi = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z ** 2)
    Phi = 0.5 * (1 + erf(alpha * z / np.sqrt(2)))
    return 2 * amplitude * phi * Phi


def _trim_to_area_fraction(time, curve, fraction=0.99):
    """Returns a mask where the cumulative area covers the given fraction of total area."""
    total_area = np.trapezoid(curve, time)
    if total_area <= 0:
        return np.ones(len(time), dtype=bool)

    cumulative = np.cumsum(0.5 * (curve[:-1] + curve[1:]) * np.diff(time))
    cumulative = np.insert(cumulative, 0, 0)

    lower = (1 - fraction) / 2 * total_area
    upper = (1 + fraction) / 2 * total_area

    return (cumulative >= lower) & (cumulative <= upper)


def plot_signal_2d(
    name: str,
    time: np.ndarray,
    time_unit: str,
    signal: np.ndarray,
    signal_unit: str,
    peaks: list[dict] | None = None,
    width: int = 520,
    height: int = 340,
    filename: str | None = None,
    dpi: int = 300,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
            mode="lines",
            line=dict(color="#4C78A8", width=1.5),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Time: %{{x:.3f}} {time_unit}<br>"
                f"Signal: %{{y:.2f}} {signal_unit}"
                "<extra></extra>"
            ),
        )
    )

    if peaks:
        colors = [
            "rgba(228, 87, 86, 0.25)",
            "rgba(114, 183, 178, 0.25)",
            "rgba(245, 133, 24, 0.25)",
            "rgba(178, 121, 162, 0.25)",
            "rgba(84, 162, 75, 0.25)",
            "rgba(238, 202, 59, 0.25)",
        ]
        border_colors = [
            "rgba(228, 87, 86, 0.6)",
            "rgba(114, 183, 178, 0.6)",
            "rgba(245, 133, 24, 0.6)",
            "rgba(178, 121, 162, 0.6)",
            "rgba(84, 162, 75, 0.6)",
            "rgba(238, 202, 59, 0.6)",
        ]

        for i, p in enumerate(peaks):
            curve = _skew_normal_curve(time, p["mu"], p["amplitude"], p["sigma"], p["alpha"])
            mask = _trim_to_area_fraction(time, curve, fraction=0.99)

            t_trimmed = time[mask]
            c_trimmed = curve[mask]

            color = colors[i % len(colors)]
            border = border_colors[i % len(border_colors)]
            label = p.get("label", f"Peak {i + 1}")

            fig.add_trace(
                go.Scatter(
                    x=t_trimmed,
                    y=c_trimmed,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=color,
                    line=dict(color=border, width=1),
                    name=label,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"Time: %{{x:.3f}} {time_unit}<br>"
                        f"Signal: %{{y:.2f}} {signal_unit}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(
            text=name,
            font=dict(size=14, color="#2c3e50"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text=f"Time ({time_unit})", font=dict(size=11)),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            gridwidth=1,
            zeroline=False,
            linecolor="#ccc",
            linewidth=1,
        ),
        yaxis=dict(
            title=dict(text=f"Signal ({signal_unit})", font=dict(size=11)),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            gridwidth=1,
            zeroline=False,
            linecolor="#ccc",
            linewidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=55, r=20, t=40, b=45),
        width=width,
        height=height,
        showlegend=bool(peaks),
        hovermode="x unified",
    )

    if filename:
        scale = dpi / 96
        fig.write_image(filename, scale=scale)

    fig.show()
    return fig