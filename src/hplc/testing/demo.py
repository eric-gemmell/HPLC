"""
Visual demos for plotting functions. Call from a REPL or notebook to eyeball output.

    from hplc.testing.demo import demo_plot_signal_2d, demo_plot_multi_graph
    demo_plot_signal_2d()
    demo_plot_multi_graph()
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from hplc.testing.mock_generators import make_raw_signal
from hplc.graphing.graph_models import TraceProperties2D, FillRegion, GraphProperties2D
from hplc.graphing import plot_2d_graph, plot_graphs


DEMO_PEAKS_190 = [
    {"mu": 120, "amplitude": 800, "sigma": 12, "alpha": 1.2},
    {"mu": 250, "amplitude": 400, "sigma": 18, "alpha": 0.5},
    {"mu": 450, "amplitude": 600, "sigma": 10, "alpha": 2.0},
]

DEMO_PEAKS_254 = [
    {"mu": 130, "amplitude": 1200, "sigma": 14, "alpha": 0.8},
    {"mu": 300, "amplitude": 900, "sigma": 20, "alpha": 1.5},
    {"mu": 500, "amplitude": 350, "sigma": 8, "alpha": 0.0},
]

DEMO_PEAKS_405 = [
    {"mu": 200, "amplitude": 500, "sigma": 25, "alpha": 0.3},
    {"mu": 400, "amplitude": 700, "sigma": 15, "alpha": 1.8},
]


def _skew_normal_curve(time, mu, amplitude, sigma, alpha):
    z = (time - mu) / sigma
    phi = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z ** 2)
    Phi = 0.5 * (1 + erf(alpha * z / np.sqrt(2)))
    return 2 * amplitude * phi * Phi


def _peaks_to_fills(time: np.ndarray, peaks: list[dict], baseline: np.ndarray) -> list[FillRegion]:
    fills = []
    for p in peaks:
        curve = _skew_normal_curve(time, p["mu"], p["amplitude"], p["sigma"], p["alpha"])
        mask = curve > 0.01 * curve.max()
        fills.append(FillRegion(
            time=time[mask],
            upper=curve[mask]+baseline[mask],
            lower=baseline[mask],
            label=f"Peak @ {p['mu']}s",
        ))
    return fills


def _make_sine_baseline(time: np.ndarray, amplitude: float = 3.0, period: float = 600.0, phase: float = 0.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * time / period + phase)


def _default_traces() -> list[TraceProperties2D]:
    traces = []
    for name, peaks, seed, phase in [
        ("DAD 190 nm", DEMO_PEAKS_190, 1, 0.0),
        ("DAD 254 nm", DEMO_PEAKS_254, 2, np.pi / 3),
        ("DAD 405 nm", DEMO_PEAKS_405, 3, np.pi / 1.5),
    ]:
        time, signal = make_raw_signal(peaks, duration=600, dt=1, noise_scale=0.5, seed=seed)
        baseline = _make_sine_baseline(time, phase=phase)
        fills = _peaks_to_fills(time, peaks, baseline)
        traces.append(TraceProperties2D(
            name=name,
            time=time,
            time_unit="s",
            signal=signal + baseline,
            signal_unit="mAU",
            fills=fills,
            baseline=baseline,
        ))
    return traces


def demo_plot_signal_2d(
    signals: list[TraceProperties2D] | None = None,
    title: str = "Demo: plot_signal_2d",
    width: int = 800,
    height: int = 500,
    filename: str | None = None,
    dpi: int = 300,
    style: str = "color",
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    plot_2d_graph(
        signals=signals or _default_traces(),
        title=title,
        width=width,
        height=height,
        filename=filename,
        dpi=dpi,
        style=style,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def demo_plot_multi_graph(
    style: str = "color",
    filename: str | None = None,
) -> None:
    """
    Visual demo for plot_graphs. Call from a REPL or notebook:

        from hplc.testing.demo import demo_plot_multi_graph
        demo_plot_multi_graph()
        demo_plot_multi_graph(style="bw")
    """
    traces = _default_traces()

    graphs = [
        GraphProperties2D(
            signals=[t],
            title=t.name,
        )
        for t in traces
    ]

    plot_graphs(
        graphs=graphs,
        cols=2,
        title="Demo: plot_graphs",
        subplot_width=520,
        subplot_height=340,
        style=style,
        filename=filename,
    )
