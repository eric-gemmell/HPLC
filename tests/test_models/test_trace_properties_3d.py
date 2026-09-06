import numpy as np
import pytest

from hplc.graphing.graph_models import TraceProperties3D


# ---------------------------------------------------------------------------
# Shared test data — a deliberately tiny cube so shapes are readable at a glance
#   3 time points x 4 wavelengths
# ---------------------------------------------------------------------------

TIME = [0.0, 1.0, 2.0]
WAVELENGTHS = [200.0, 210.0, 220.0, 230.0]
SIGNAL = [
    [0.0, 1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0, 7.0],
    [8.0, 9.0, 10.0, 11.0],
]
COLORS = ["#FFFFFF", "#4C78A8", "#000000"]
COLOR_VALUES = [0.0, 5.0, 11.0]


def make_props(**overrides):
    kwargs = dict(
        name="test_trace",
        time=TIME,
        time_unit="s",
        wavelengths=WAVELENGTHS,
        wavelength_unit="nm",
        signal=SIGNAL,
        signal_unit="mAU",
    )
    kwargs.update(overrides)
    return TraceProperties3D(**kwargs)


@pytest.fixture
def props():
    return make_props()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_stores_every_field(props):
    assert props.name == "test_trace"
    assert props.time_unit == "s"
    assert props.wavelength_unit == "nm"
    assert props.signal_unit == "mAU"
    np.testing.assert_array_equal(props.time, TIME)
    np.testing.assert_array_equal(props.wavelengths, WAVELENGTHS)
    np.testing.assert_array_equal(props.signal, SIGNAL)


def test_colors_and_color_values_default_to_none(props):
    assert props.colors is None
    assert props.color_values is None


def test_stores_colors_and_color_values():
    props = make_props(colors=COLORS, color_values=COLOR_VALUES)
    assert props.colors == COLORS
    np.testing.assert_array_equal(props.color_values, COLOR_VALUES)


# ---------------------------------------------------------------------------
# Coercion — lists in, ndarrays out
# ---------------------------------------------------------------------------

def test_list_inputs_are_coerced_to_ndarray(props):
    assert isinstance(props.time, np.ndarray)
    assert isinstance(props.wavelengths, np.ndarray)
    assert isinstance(props.signal, np.ndarray)


def test_color_values_coerced_to_ndarray():
    props = make_props(colors=COLORS, color_values=COLOR_VALUES)
    assert isinstance(props.color_values, np.ndarray)


def test_ndarray_inputs_are_accepted():
    props = make_props(
        time=np.array(TIME),
        wavelengths=np.array(WAVELENGTHS),
        signal=np.array(SIGNAL),
    )
    np.testing.assert_array_equal(props.signal, SIGNAL)


# ---------------------------------------------------------------------------
# Signal shape — signal is (len(time), len(wavelengths))
# ---------------------------------------------------------------------------

def test_signal_is_indexed_time_first(props):
    assert props.signal.shape == (len(TIME), len(WAVELENGTHS))
    np.testing.assert_array_equal(props.signal[0], [0.0, 1.0, 2.0, 3.0])


def test_transposed_signal_raises():
    with pytest.raises(ValueError, match="[Ss]hape"):
        make_props(signal=np.array(SIGNAL).T)


def test_one_dimensional_signal_raises():
    with pytest.raises(ValueError, match="2D|two|dimension"):
        make_props(signal=[1.0, 2.0, 3.0])


def test_three_dimensional_signal_raises():
    with pytest.raises(ValueError, match="2D|two|dimension"):
        make_props(signal=np.zeros((3, 4, 2)))


def test_signal_with_wrong_number_of_time_rows_raises():
    with pytest.raises(ValueError, match="[Ss]hape"):
        make_props(signal=np.zeros((5, len(WAVELENGTHS))))


def test_signal_with_wrong_number_of_wavelength_columns_raises():
    with pytest.raises(ValueError, match="[Ss]hape"):
        make_props(signal=np.zeros((len(TIME), 7)))


# ---------------------------------------------------------------------------
# Empty axes
# ---------------------------------------------------------------------------

def test_empty_time_raises():
    with pytest.raises(ValueError, match="[Tt]ime"):
        make_props(time=[], signal=np.zeros((0, len(WAVELENGTHS))))


def test_empty_wavelengths_raises():
    with pytest.raises(ValueError, match="[Ww]avelength"):
        make_props(wavelengths=[], signal=np.zeros((len(TIME), 0)))


# ---------------------------------------------------------------------------
# Colour validation
# ---------------------------------------------------------------------------

def test_single_colour_raises():
    with pytest.raises(ValueError, match="[Cc]olor"):
        make_props(colors=["#FFFFFF"], color_values=[0.0])


def test_empty_colours_raises():
    with pytest.raises(ValueError, match="[Cc]olor"):
        make_props(colors=[], color_values=[])


def test_two_colours_is_allowed():
    props = make_props(colors=["#FFFFFF", "#000000"], color_values=[0.0, 11.0])
    assert len(props.colors) == 2


def test_mismatched_colour_value_length_raises():
    with pytest.raises(ValueError, match="[Cc]olor"):
        make_props(colors=COLORS, color_values=[0.0, 11.0])


def test_colours_without_colour_values_raises():
    with pytest.raises(ValueError, match="[Cc]olor"):
        make_props(colors=COLORS)


def test_colour_values_without_colours_raises():
    with pytest.raises(ValueError, match="[Cc]olor"):
        make_props(color_values=COLOR_VALUES)


# ---------------------------------------------------------------------------
# Equality — numpy-safe, and false when any field differs
# ---------------------------------------------------------------------------

def test_identical_properties_are_equal():
    assert make_props() == make_props()


def test_identical_properties_with_colours_are_equal():
    a = make_props(colors=COLORS, color_values=COLOR_VALUES)
    b = make_props(colors=COLORS, color_values=COLOR_VALUES)
    assert a == b


@pytest.mark.parametrize("field, value", [
    ("name", "a_different_name"),
    ("time", [0.0, 1.0, 99.0]),
    ("time_unit", "min"),
    ("wavelengths", [200.0, 210.0, 220.0, 999.0]),
    ("wavelength_unit", "um"),
    ("signal", [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 99.0]]),
    ("signal_unit", "AU"),
])
def test_differing_field_makes_properties_unequal(field, value):
    assert make_props() != make_props(**{field: value})


def test_differing_colours_make_properties_unequal():
    a = make_props(colors=COLORS, color_values=COLOR_VALUES)
    b = make_props(colors=["#FFFFFF", "#4C78A8", "#FF0000"], color_values=COLOR_VALUES)
    assert a != b


def test_differing_colour_values_make_properties_unequal():
    a = make_props(colors=COLORS, color_values=COLOR_VALUES)
    b = make_props(colors=COLORS, color_values=[0.0, 6.0, 11.0])
    assert a != b


def test_colours_set_versus_unset_are_unequal():
    assert make_props() != make_props(colors=COLORS, color_values=COLOR_VALUES)


def test_not_equal_to_other_types(props):
    assert props != "not a trace"
    assert props != 42
