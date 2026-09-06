"""
Visual demos for plotting functions. Call from a REPL or notebook to eyeball output.

    from hplc.testing.demo import demo_plot_signal_2d, demo_plot_multi_graph, demo_plot_signal_3d
    demo_plot_signal_2d()
    demo_plot_multi_graph()
    demo_plot_signal_3d()
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from hplc.testing.mock_generators import make_raw_signal, make_signal3d
from hplc.graphing.graph_models import TraceProperties2D, TraceProperties3D, FillRegion, GraphProperties2D
from hplc.graphing import plot_2d_graph, plot_3d_graph, plot_graphs
from hplc.models import Chromatogram, Signal2D


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


# Each analyte: a chromatographic peak in time x a UV absorption band in wavelength.
DEMO_ANALYTES_3D = [
    # rt (s), chromatographic sigma/skew, lambda_max (nm), band width (nm), strength
    {"name": "pNPS",     "mu": 120, "sigma": 10, "alpha": 1.0, "lmax": 300, "bw": 34, "amp": 900},
    {"name": "pNP",      "mu": 250, "sigma": 14, "alpha": 2.5, "lmax": 400, "bw": 44, "amp": 1400},
    {"name": "impurity", "mu": 330, "sigma": 8,  "alpha": 0.4, "lmax": 254, "bw": 22, "amp": 450},
    {"name": "late tail","mu": 470, "sigma": 26, "alpha": 6.0, "lmax": 228, "bw": 30, "amp": 700},
]


def _make_pda_cube(
    duration: float = 600.0,
    dt: float = 1.0,
    wl_min: float = 190.0,
    wl_max: float = 600.0,
    wl_step: float = 2.0,
    noise_scale: float = 4.0,
    seed: int = 7,
):
    """Build a synthetic (time x wavelength) PDA cube.

    Returns (time, wavelengths, signal) with signal.shape == (len(time), len(wavelengths)).
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0, duration, dt, dtype=float)
    wavelengths = np.arange(wl_min, wl_max + wl_step, wl_step, dtype=float)

    signal = rng.normal(0.0, noise_scale, size=(len(time), len(wavelengths)))

    # Solvent absorbs hard at low wavelength — a standing ridge down the 190 nm edge.
    signal += 260.0 * np.exp(-(wavelengths[None, :] - 190.0) / 26.0)

    for a in DEMO_ANALYTES_3D:
        chrom = _skew_normal_curve(time, a["mu"], a["amp"], a["sigma"], a["alpha"])
        spectrum = np.exp(-0.5 * ((wavelengths - a["lmax"]) / a["bw"]) ** 2)
        signal += np.outer(chrom, spectrum)

    return time, wavelengths, signal


def _default_trace_3d(name: str = "PDA 190-600 nm") -> TraceProperties3D:
    time, wavelengths, signal = _make_pda_cube()
    return TraceProperties3D(
        name=name,
        time=time,
        time_unit="s",
        wavelengths=wavelengths,
        wavelength_unit="nm",
        signal=signal,
        signal_unit="mAU",
        # Absolute colour stops, in mAU — not normalised, so the mapping is
        # comparable between runs.
        colors=["#FFFFFF", "#D9E3F0", "#8FB8DE", "#4C78A8", "#E45756", "#7A1416"],
        color_values=np.array([0.0, 20.0, 60.0, 140.0, 260.0, 420.0]),
    )


def demo_plot_signal_3d(
    signal: TraceProperties3D | None = None,
    title: str = "Demo: plot_3d_graph",
    width: int = 900,
    height: int = 560,
    filename: str | None = None,
    dpi: int = 300,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
):
    """
    Visual demo for plot_3d_graph. Call from a REPL or notebook:

        from hplc.testing.demo import demo_plot_signal_3d
        demo_plot_signal_3d()

        # zoom in on the pNP band
        demo_plot_signal_3d(x_min=200, x_max=320, y_min=340, y_max=460)

        # clip the colour scale to bring the small impurity out
        demo_plot_signal_3d(z_max=120)

    Returns the go.Figure so you can keep poking at it.
    """
    return plot_3d_graph(
        signal=signal if signal is not None else _default_trace_3d(),
        title=title,
        width=width,
        height=height,
        filename=filename,
        dpi=dpi,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )


def make_demo_chromatogram_3d(filename: str = "demo_sample.lcd") -> Chromatogram:
    """A mock Shimadzu-shaped chromatogram: one PDA cube plus one RID trace."""
    analytes = [
        {"mu": 120, "amplitude": 900,  "sigma": 10, "alpha": 1.0, "lmax": 300, "band": 34},
        {"mu": 250, "amplitude": 1400, "sigma": 14, "alpha": 2.5, "lmax": 400, "band": 44},
        {"mu": 330, "amplitude": 450,  "sigma": 8,  "alpha": 0.4, "lmax": 254, "band": 22},
        {"mu": 470, "amplitude": 700,  "sigma": 26, "alpha": 6.0, "lmax": 228, "band": 30},
    ]
    pda = make_signal3d(
        analytes,
        detector_name="PDA",
        filename=filename,
        duration=600,
        dt=1,
        noise_scale=4.0,
        seed=7,
    )

    # A refractive-index trace alongside it, as a real .lcd carries.
    rid_time, rid_signal = make_raw_signal(
        [{"mu": a["mu"], "amplitude": a["amplitude"] / 3, "sigma": a["sigma"], "alpha": a["alpha"]}
         for a in analytes],
        duration=600, dt=1, noise_scale=3.0, seed=11,
    )
    rid = Signal2D(
        detector_name="RID",
        time=rid_time,
        time_unit="s",
        signal=rid_signal,
        signal_unit="uRI",
        filename=filename,
    )

    return Chromatogram(signals={"PDA": pda, "RID": rid}, filename=filename)


def demo_chromatogram_3d(wavelengths=(254, 300, 400)):
    """
    Visual demo of a mixed 2D/3D chromatogram. Call from a REPL or notebook:

        from hplc.testing.demo import demo_chromatogram_3d
        demo_chromatogram_3d()

    Shows, in order:
      1. the PDA cube and the RID trace side by side in one figure
      2. the same chromatogram after extracting three single-wavelength traces

    Returns the extracted Chromatogram so you can keep working with it.
    """
    chrom = make_demo_chromatogram_3d()
    chrom.display(title="Mock .lcd — PDA cube + RID trace", width=560, height=380)

    extracted = chrom.extract_wavelengths(list(wavelengths), bandwidth=4)
    extracted.display(
        title=f"After extract_wavelengths({list(wavelengths)}, bandwidth=4)",
        width=460,
        height=320,
    )
    return extracted
