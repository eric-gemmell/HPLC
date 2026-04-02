import pytest
from hplc.models import Peak2D

def test_peak2d_repr():
    peak = Peak2D(
        retention_time=200.0,
        max_height=100.0,
        area=1234.5,
        start_time=185.0,
        end_time=220.0,
        unit="mAU",
    )
    r = repr(peak)
    assert "Peak2D" in r
    assert "200.0" in r
    assert "100.0" in r
    assert "1234.5" in r
    assert "185.0" in r
    assert "220.0" in r
    assert "mAU" in r