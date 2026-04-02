from __future__ import annotations

import pytest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unittest.mock import patch

from HPLC.graphing import plot_signal_2d


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(go.Figure, "show", lambda self: None)


def _sig(name="UV 254 nm", **overrides):
    d = {
        "name": name,
        "time": np.linspace(0, 10, 50),
        "time_unit": "min",
        "signal": np.sin(np.linspace(0, 2 * np.pi, 50)),
        "signal_unit": "mAU",
    }
    d.update(overrides)
    return d


class TestAcceptsParameters:

    def test_single_signal(self):
        fig = plot_signal_2d(signals=[_sig()])
        assert isinstance(fig, go.Figure)

    def test_multiple_signals(self):
        fig = plot_signal_2d(signals=[_sig("A"), _sig("B")])
        assert isinstance(fig, go.Figure)

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError):
            plot_signal_2d(signals=[])

    def test_title(self):
        fig = plot_signal_2d(signals=[_sig()], title="My Title")
        assert isinstance(fig, go.Figure)

    def test_width_height_dpi(self):
        fig = plot_signal_2d(signals=[_sig()], width=800, height=600, dpi=150)
        assert isinstance(fig, go.Figure)

    def test_style(self):
        fig = plot_signal_2d(signals=[_sig()], style="color")
        assert isinstance(fig, go.Figure)

    def test_x_lim_and_y_lim(self):
        fig = plot_signal_2d(signals=[_sig()], x_lim=(0, 5), y_lim=(-1, 1))
        assert isinstance(fig, go.Figure)

    def test_x_lim_and_y_lim_accept_partial_none(self):
        fig = plot_signal_2d(signals=[_sig()], x_lim=(None, 5), y_lim=(-1, None))
        assert isinstance(fig, go.Figure)

    def test_signal_optional_fields(self):
        """All optional per-signal fields are accepted without error."""
        fig = plot_signal_2d(signals=[_sig(
            peaks=[{"mu": 5.0, "amplitude": 1.0, "sigma": 0.5, "alpha": 0.0}],
            delta=10.0,
            stretch_factor=2.0,
            y_offset=50.0,
            scale=3.0,
            color="#FF0000",
        )])
        assert isinstance(fig, go.Figure)


class TestFigureInjection:

    def test_creates_figure_when_none(self):
        fig = plot_signal_2d(signals=[_sig()])
        assert isinstance(fig, go.Figure)

    def test_draws_onto_existing_figure(self):
        existing = make_subplots(rows=1, cols=2)
        fig = plot_signal_2d(signals=[_sig()], fig=existing, row=1, col=2)
        assert fig is existing
        assert len(fig.data) > 0

    def test_does_not_show_when_fig_provided(self):
        existing = make_subplots(rows=1, cols=1)
        with patch.object(go.Figure, "show") as mock_show:
            plot_signal_2d(signals=[_sig()], fig=existing, row=1, col=1)
            mock_show.assert_not_called()

    def test_shows_when_standalone(self):
        with patch.object(go.Figure, "show") as mock_show:
            plot_signal_2d(signals=[_sig()])
            mock_show.assert_called_once()
