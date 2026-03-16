import pytest
from pathlib import Path

from hplc.core.chromatogram import Chromatogram
from hplc.io import load


EXAMPLE_FILE = (
    Path(__file__).parent.parent
    / "example_files"
    / "EGG_HILIC_NaCl_0.2mM_20ul_80%MeCN_pH3_20mM_NH4OHC13.03.2026 19-13-41.dat"
)


class TestChromatogramDisplay:
    """Test Chromatogram.display() via dependency injection."""

    def test_display_passes_self_and_returns_self(self):
        chrom = load(str(EXAMPLE_FILE))
        received = []
        Chromatogram.register_display(lambda c: received.append(c))
        try:
            result = chrom.display()
            assert result is chrom
            assert received[0] is chrom
        finally:
            Chromatogram.register_display(None)

    def test_display_raises_when_no_fn_registered(self):
        chrom = load(str(EXAMPLE_FILE))
        Chromatogram.register_display(None)
        with pytest.raises(RuntimeError, match="No display function registered"):
            chrom.display()

    def test_display_fn_receives_signals_and_metadata(self):
        chrom = load(str(EXAMPLE_FILE))
        received = []
        Chromatogram.register_display(lambda c: received.append(c))
        try:
            chrom.display()
            displayed = received[0]
            assert len(displayed.signals) == 5
            assert "sample_name" in displayed.metadata
        finally:
            Chromatogram.register_display(None)
