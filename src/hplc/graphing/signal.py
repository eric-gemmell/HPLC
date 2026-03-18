import plotly.graph_objects as go
import numpy as np


def plot_signal_2d(
    name: str,
    time: np.ndarray,
    time_unit: str,
    signal: np.ndarray,
    signal_unit: str,
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
        showlegend=False,
        hovermode="x unified",
    )

    if filename:
        scale = dpi / 96
        fig.write_image(filename, scale=scale)

    fig.show()
    return fig