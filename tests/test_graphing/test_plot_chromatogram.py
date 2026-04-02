from __future__ import annotations

import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import patch

from HPLC.graphing import plot_chromatogram


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(go.Figure, "show", lambda self: None)


def _signal_data():
    return {
        "time": np.linspace(0, 10, 50),
        "time_unit": "min",
        "signal": np.sin(np.linspace(0, 2 * np.pi, 50)),
        "signal_unit": "mAU",
    }


class TestPlotChromatogramUsesPlotSignal2d:

    def test_calls_plot_signal_2d(self):
        signals = {"UV": _signal_data(), "FLD": _signal_data()}
        with patch("HPLC.graphing.chromatogram.plot_signal_2d") as mock:
            mock.return_value = go.Figure()
            plot_chromatogram(signals=signals)
            assert mock.call_count == len(signals)

    def test_single_signal_calls_plot_signal_2d_once(self):
        signals = {"UV": _signal_data()}
        with patch("HPLC.graphing.chromatogram.plot_signal_2d") as mock:
            mock.return_value = go.Figure()
            plot_chromatogram(signals=signals)
            mock.assert_called_once()

    def test_returns_figure(self):
        signals = {"UV": _signal_data()}
        with patch("HPLC.graphing.chromatogram.plot_signal_2d") as mock:
            mock.return_value = go.Figure()
            fig = plot_chromatogram(signals=signals)
            assert isinstance(fig, go.Figure)
