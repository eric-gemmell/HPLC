import plotly.graph_objects as go

from hplc.core.signal import Signal2D


def _format_metadata(metadata: dict) -> str:
    if not metadata:
        return ""
    lines = [f"<b>Metadata:</b>"]
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    return "<br>".join(lines)


def plot_signal2d(signal: Signal2D) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=signal.time,
            y=signal.signal,
            mode="lines",
            name=signal.detector_name,
            line=dict(color="steelblue", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(70, 130, 180, 0.15)",
            hovertemplate="Time: %{x:.2f} min<br>Signal: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{signal.detector_name}</b>",
            font=dict(size=18),
        ),
        xaxis=dict(
            title=signal.time_unit,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=f"Signal {measurement_unit}",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        hovermode="x unified",
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=30, t=60, b=50),
        plot_bgcolor="white",
    )

    fig.show()
    return fig
