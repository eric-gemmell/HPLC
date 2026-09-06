import numpy as np
import pytest
from unittest.mock import patch

from hplc.models import Signal2D, Signal3D
from hplc.models.signal import Signal
from hplc.models.signal_stack import SignalStack2D
from hplc.graphing.graph_models import TraceProperties3D


# ---------------------------------------------------------------------------
# A tiny deterministic cube: 3 time points x 4 wavelengths
#
#   wavelength ->  200  210  220  230
#   time 0.0        0    1    2    3
#   time 1.0        4    5    6    7
#   time 2.0        8    9   10   11
# ---------------------------------------------------------------------------

TIME = [0.0, 1.0, 2.0]
WAVELENGTHS = [200.0, 210.0, 220.0, 230.0]
SIGNAL = [
    [0.0, 1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0, 7.0],
    [8.0, 9.0, 10.0, 11.0],
]


def make_signal_3d(**overrides):
    kwargs = dict(
        filename="test_signal",
        detector_name="PDA",
        time=TIME,
        time_unit="s",
        wavelengths=WAVELENGTHS,
        wavelength_unit="nm",
        signal=SIGNAL,
        signal_unit="mAU",
    )
    kwargs.update(overrides)
    return Signal3D(**kwargs)


@pytest.fixture
def sig():
    return make_signal_3d()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_is_a_signal(sig):
    assert isinstance(sig, Signal)


def test_construction_stores_fields(sig):
    assert sig.filename == "test_signal"
    assert sig.detector_name == "PDA"
    assert sig.time_unit == "s"
    assert sig.wavelength_unit == "nm"
    assert sig.signal_unit == "mAU"
    np.testing.assert_array_equal(sig.time, TIME)
    np.testing.assert_array_equal(sig.wavelengths, WAVELENGTHS)
    np.testing.assert_array_equal(sig.signal, SIGNAL)


def test_list_inputs_are_coerced_to_ndarray(sig):
    assert isinstance(sig.time, np.ndarray)
    assert isinstance(sig.wavelengths, np.ndarray)
    assert isinstance(sig.signal, np.ndarray)


def test_len_is_number_of_time_points(sig):
    assert len(sig) == 3


def test_signal_is_indexed_time_first(sig):
    assert sig.signal.shape == (len(TIME), len(WAVELENGTHS))
    np.testing.assert_array_equal(sig.signal[0], [0.0, 1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_empty_detector_name_raises():
    with pytest.raises(ValueError, match="Detector name"):
        make_signal_3d(detector_name="")


def test_empty_time_raises():
    with pytest.raises(ValueError, match="at least one"):
        make_signal_3d(time=[], signal=np.zeros((0, 4)))


def test_empty_wavelengths_raises():
    with pytest.raises(ValueError, match="[Ww]avelength"):
        make_signal_3d(wavelengths=[], signal=np.zeros((3, 0)))


def test_empty_wavelength_unit_raises():
    with pytest.raises(ValueError, match="[Ww]avelength unit"):
        make_signal_3d(wavelength_unit="")


def test_one_dimensional_signal_raises():
    with pytest.raises(ValueError, match="2D|two|dimension"):
        make_signal_3d(signal=[1.0, 2.0, 3.0])


def test_transposed_signal_raises():
    with pytest.raises(ValueError, match="[Ss]hape"):
        make_signal_3d(signal=np.array(SIGNAL).T)


def test_signal_with_wrong_shape_raises():
    with pytest.raises(ValueError, match="[Ss]hape"):
        make_signal_3d(signal=np.zeros((5, 4)))


# ---------------------------------------------------------------------------
# to_trace_properties()
# ---------------------------------------------------------------------------

def test_to_trace_properties_returns_trace_properties_3d(sig):
    assert isinstance(sig.to_trace_properties(), TraceProperties3D)


def test_to_trace_properties_instantiated_correctly(sig):
    trace = sig.to_trace_properties()
    assert trace.name == f"{sig.filename}_{sig.detector_name}"
    np.testing.assert_array_equal(trace.time, sig.time)
    np.testing.assert_array_equal(trace.wavelengths, sig.wavelengths)
    np.testing.assert_array_equal(trace.signal, sig.signal)
    assert trace.time_unit == sig.time_unit
    assert trace.wavelength_unit == sig.wavelength_unit
    assert trace.signal_unit == sig.signal_unit


def test_to_trace_properties_has_no_colours_by_default(sig):
    trace = sig.to_trace_properties()
    assert trace.colors is None
    assert trace.color_values is None


def test_to_trace_properties_accepts_colours(sig):
    colors = ["#FFFFFF", "#000000"]
    color_values = [0.0, 11.0]
    trace = sig.to_trace_properties(colors=colors, color_values=color_values)
    assert trace.colors == colors
    np.testing.assert_array_equal(trace.color_values, color_values)


# ---------------------------------------------------------------------------
# display()
# ---------------------------------------------------------------------------

@patch("hplc.models.signal.plot_3d_graph")
def test_display_returns_self(mock_plot, sig):
    assert sig.display() is sig


@patch("hplc.models.signal.plot_3d_graph")
def test_display_calls_plot_3d_graph_with_signal_data(mock_plot, sig):
    sig.display()
    trace = sig.to_trace_properties()
    mock_plot.assert_called_once_with(
        signal=trace,
        title=trace.name,
        width=520,
        height=340,
        filename=None,
        dpi=300,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        z_min=None,
        z_max=None,
        fig=None,
    )


@patch("hplc.models.signal.plot_3d_graph")
def test_display_accepts_optional_parameters(mock_plot, sig):
    colors = ["#FFFFFF", "#000000"]
    color_values = [0.0, 11.0]
    sig.display(
        width=800, height=600, filename="out.png", dpi=150, title="My PDA",
        x_min=0, x_max=2, y_min=200, y_max=230, z_min=0, z_max=11,
        colors=colors, color_values=color_values,
    )
    mock_plot.assert_called_once_with(
        signal=sig.to_trace_properties(colors=colors, color_values=color_values),
        title="My PDA",
        width=800,
        height=600,
        filename="out.png",
        dpi=150,
        x_min=0,
        x_max=2,
        y_min=200,
        y_max=230,
        z_min=0,
        z_max=11,
        fig=None,
    )


# ---------------------------------------------------------------------------
# at_wavelength()
# ---------------------------------------------------------------------------

def test_at_wavelength_returns_signal_2d(sig):
    assert isinstance(sig.at_wavelength(210), Signal2D)


def test_at_wavelength_extracts_exact_column(sig):
    extracted = sig.at_wavelength(210)
    np.testing.assert_array_equal(extracted.signal, [1.0, 5.0, 9.0])


def test_at_wavelength_extracts_first_and_last_columns(sig):
    np.testing.assert_array_equal(sig.at_wavelength(200).signal, [0.0, 4.0, 8.0])
    np.testing.assert_array_equal(sig.at_wavelength(230).signal, [3.0, 7.0, 11.0])


def test_at_wavelength_snaps_to_nearest_column(sig):
    extracted = sig.at_wavelength(214)
    np.testing.assert_array_equal(extracted.signal, [1.0, 5.0, 9.0])


def test_at_wavelength_preserves_time_axis(sig):
    extracted = sig.at_wavelength(210)
    np.testing.assert_array_equal(extracted.time, sig.time)
    assert extracted.time_unit == sig.time_unit
    assert extracted.signal_unit == sig.signal_unit
    assert extracted.filename == sig.filename


def test_at_wavelength_names_the_detector_with_the_wavelength(sig):
    extracted = sig.at_wavelength(210)
    assert "210" in extracted.detector_name
    assert "nm" in extracted.detector_name


def test_at_wavelength_records_the_wavelength_used(sig):
    extracted = sig.at_wavelength(214)
    assert extracted.additional_data["wavelength"] == 210.0


def test_at_wavelength_below_range_raises(sig):
    with pytest.raises(ValueError, match="[Ww]avelength"):
        sig.at_wavelength(150)


def test_at_wavelength_above_range_raises(sig):
    with pytest.raises(ValueError, match="[Ww]avelength"):
        sig.at_wavelength(400)


def test_at_wavelength_with_bandwidth_averages_columns(sig):
    # window [205, 225] covers the 210 and 220 columns
    extracted = sig.at_wavelength(215, bandwidth=20)
    np.testing.assert_allclose(extracted.signal, [1.5, 5.5, 9.5])


def test_at_wavelength_with_zero_bandwidth_is_a_single_column(sig):
    extracted = sig.at_wavelength(210, bandwidth=0)
    np.testing.assert_array_equal(extracted.signal, [1.0, 5.0, 9.0])


def test_at_wavelength_records_the_bandwidth_used(sig):
    extracted = sig.at_wavelength(215, bandwidth=20)
    assert extracted.additional_data["bandwidth"] == 20


def test_at_wavelength_negative_bandwidth_raises(sig):
    with pytest.raises(ValueError, match="[Bb]andwidth"):
        sig.at_wavelength(210, bandwidth=-5)


# ---------------------------------------------------------------------------
# at_wavelengths()
# ---------------------------------------------------------------------------

def test_at_wavelengths_returns_signal_stack_2d(sig):
    assert isinstance(sig.at_wavelengths([200, 220]), SignalStack2D)


def test_at_wavelengths_returns_one_signal_per_wavelength(sig):
    stack = sig.at_wavelengths([200, 210, 220])
    assert len(stack) == 3
    np.testing.assert_array_equal(stack.signals[0].signal, [0.0, 4.0, 8.0])
    np.testing.assert_array_equal(stack.signals[1].signal, [1.0, 5.0, 9.0])
    np.testing.assert_array_equal(stack.signals[2].signal, [2.0, 6.0, 10.0])


def test_at_wavelengths_rejects_an_out_of_range_wavelength(sig):
    with pytest.raises(ValueError, match="[Ww]avelength"):
        sig.at_wavelengths([200, 999])


def test_at_wavelengths_rejects_empty_list(sig):
    with pytest.raises(ValueError, match="at least one"):
        sig.at_wavelengths([])


# ---------------------------------------------------------------------------
# max_plot()
# ---------------------------------------------------------------------------

def test_max_plot_returns_signal_2d(sig):
    assert isinstance(sig.max_plot(), Signal2D)


def test_max_plot_takes_the_maximum_across_wavelengths(sig):
    np.testing.assert_array_equal(sig.max_plot().signal, [3.0, 7.0, 11.0])


def test_max_plot_preserves_time_axis(sig):
    plot = sig.max_plot()
    np.testing.assert_array_equal(plot.time, sig.time)
    assert plot.signal_unit == sig.signal_unit


def test_max_plot_names_the_detector(sig):
    assert "max" in sig.max_plot().detector_name.lower()
