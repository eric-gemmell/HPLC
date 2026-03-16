import numpy as np
import pytest
from pathlib import Path

from hplc.core.signal import Signal2D
from hplc.io import load


EXAMPLE_FILE = (
    Path(__file__).parent.parent
    / "example_files"
    / "EGG_HILIC_NaCl_0.2mM_20ul_80%MeCN_pH3_20mM_NH4OHC13.03.2026 19-13-41.dat"
)


class TestSignal2DDisplay:
    """Test Signal2D.display() via dependency injection."""

    def test_display_passes_self_and_returns_self(self):
        sig = Signal2D(
            time=np.array([0.0, 0.1, 0.2]),
            signal=np.array([10.0, 20.0, 30.0]),
            detector_name="UV",
            time_unit="mins",
            measurement_unit="mAU",
            metadata={"detector_id": 1},
        )
        received = []
        Signal2D.register_display(lambda s: received.append(s))
        try:
            result = sig.display()
            assert result is sig
            assert received[0] is sig
        finally:
            Signal2D.register_display(None)

    def test_display_raises_when_no_fn_registered(self):
        sig = Signal2D(
            time=np.array([0.0, 0.1]),
            signal=np.array([1.0, 2.0]),
            detector_name="UV",
            time_unit="mins",
            measurement_unit="mAU",
        )
        Signal2D.register_display(None)
        with pytest.raises(RuntimeError, match="No display function registered"):
            sig.display()

    def test_display_on_loaded_signals(self):
        """Load a real file, inject a fake display, verify all signals dispatched."""
        chrom = load(str(EXAMPLE_FILE))
        received = []
        Signal2D.register_display(lambda s: received.append(s))
        try:
            for sig in chrom.signals.values():
                sig.display()

            assert len(received) == len(chrom.signals)
            for sig in received:
                assert isinstance(sig, Signal2D)
                assert len(sig.time) > 0
                assert len(sig.signal) == len(sig.time)
                assert sig.detector_name
                assert sig.time_unit == "mins"
                assert sig.measurement_unit
        finally:
            Signal2D.register_display(None)
