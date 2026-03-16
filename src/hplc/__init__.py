"""HPLC — High-Performance Liquid Chromatography analysis library."""

from hplc.core.signal import Signal, Signal2D
from hplc.core.chromatogram import Chromatogram
from hplc.core.peak import Peak
from hplc.io import load


from hplc.display.interactive import plot_signal2d
import plotly.io as pio

pio.renderers.default = "iframe"

# Wire display functions into signal types
Signal2D.register_display(plot_signal2d)
# Signal3D.register_display(plot_signal3d)

__all__ = [
    "Signal",
    "Signal2D",
    # "Signal3D",
    "Chromatogram",
    "Peak",
    "load",
]
