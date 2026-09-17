import tempfile
import subprocess
import sys
from pathlib import Path
import numpy as np
from matplotlib import colormaps
import hyperscope as notebook_api
from hyperscope import (
    APP_VERSION,
    CURVE_CURSOR_COLORS,
    PLOT_EXPORT_FILETYPES,
    DataSet,
    Detection,
    collapse_spectral_range,
    contrast_reference_limits,
    consume_first_use,
    curve_cursor_statistics,
    curve_axis_scale,
    curve_range_dataset,
    curve_sidebar_information,
    curve_style_from_controls,
    data_export_filetypes,
    dataset_with_replaced_axis,
    default_curve_style,
    default_fit_style,
    detect_text,
    display_colormap,
    export_dataset_text,
    fit_curve,
    fit_curve_selection,
    fit_sidebar_information,
    grouped_colormaps,
    hyperspectral_sections,
    hyperspectral_slice_dataset,
    image_contrast_limits,
    image_line_profile,
    image_profile_dataset,
    initial_hyperspectral_position,
    labeled_fit_parameters,
    load_text,
    metadata_sidebar_text,
    nearest_axis_index,
    normalize_display_channel,
    orient_image_for_export,
    orient_spectral_planes,
    profile_cursor_visibility,
    raster_export_array,
    save_display_image,
    duplicate_dataset,
    validate_session_payload,
)
from hyperscope import (
    CurveNotebookView,
    HyperspectralNotebookView,
    NotebookView,
    coordinate_slider_options,
    dataset_from_array,
    import_data as notebook_import_data,
    import_numpy,
)

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"


def test_hyperscope_import_is_tk_free():
    script = """
import builtins
import sys

original_import = builtins.__import__
def reject_tk(name, *args, **kwargs):
    if name == 'tkinter' or name.startswith('tkinter.'):
        raise AssertionError(f'unexpected desktop dependency: {name}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_tk
import hyperscope
assert callable(hyperscope.open_view)
assert 'tkinter' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_desktop_launcher_has_no_jupyter_dependency():
    source = (ROOT / "hyperscope_desktop.py").read_text()
    assert "hyperscope_jupyter" not in source
    assert "import ipywidgets" not in source
    assert "from hyperscope import *" in source

def test_samples():
    expected = {
        "ramanSpecEV.txt": "curve",
        "ramanSpecLambda.txt": "curve",
        "ramanSpecRshift.txt": "curve",
        "ramanSpecRshiftY.txt": "curve",
        "afmImage.txt": "image",
        "ramanMap.txt": "hyperspectral",
        "tersMap.txt": "hyperspectral",
    }
    for name, kind in expected.items():
        ds = load_text(str(SAMPLES/name), detect_text(str(SAMPLES/name))[0])
        assert ds.kind == kind and ds.data.size

def test_fit_and_line():
    x=np.linspace(-2,2,100); y=3*x+2
    fitted,p=fit_curve(x,y,"line")
    assert np.max(abs(fitted-y)) < 1e-8
    with tempfile.TemporaryDirectory() as d:
        pth=Path(d)/"line.txt";np.savetxt(pth,[np.arange(12)])
        detection = Detection("curve", 1, (12,), "", "row_line")
        ds = load_text(str(pth), detection)
        assert ds.data.shape==(12,)
        assert ds.kind == "curve"
    row_path = SAMPLES / "hyperspectral_single_line.txt"
    row_detection = detect_text(str(row_path))[0]
    assert row_detection.kind == "curve"
    assert row_detection.parser == "row_line"

def test_hyperspectral_sections_follow_cursor():
    # A unique place value for each axis makes accidental axis swaps obvious.
    data = np.fromfunction(lambda y, x, z: 100*y + 10*x + z, (4, 5, 6))
    sections = hyperspectral_sections(data, ix=3, iy=2, iz=4)
    np.testing.assert_array_equal(sections["xy"], data[:, :, 4])
    np.testing.assert_array_equal(sections["xz"], data[2, :, :].T)
    np.testing.assert_array_equal(sections["yz"], data[:, 3, :])
    np.testing.assert_array_equal(sections["x_profile"], data[2, :, 4])
    np.testing.assert_array_equal(sections["y_profile"], data[:, 3, 4])
    np.testing.assert_array_equal(sections["z_profile"], data[2, 3, :])
    assert sections["z_profile"][4] == sections["xy"][2, 3]
    assert sections["x_profile"][3] == sections["xy"][2, 3]
    assert sections["y_profile"][2] == sections["xy"][2, 3]

def test_contrast_reference_is_inclusive_and_tracks_z():
    data = np.fromfunction(lambda y, x, z: 100*y + 10*x + z, (4, 5, 6))
    reference = [[1, 1], [3, 2]]
    assert contrast_reference_limits(data, 0, reference) == (110.0, 230.0)
    assert contrast_reference_limits(data, 5, reference) == (115.0, 235.0)

def test_rgb_normalization_uses_or_ignores_contrast_reference():
    values = np.array([
        [0.0, 10.0, 20.0],
        [30.0, 40.0, 50.0],
        [60.0, 70.0, 80.0],
    ])
    reference = [[1, 0], [2, 1]]  # Values 10, 20, 40, 50.
    referenced = normalize_display_channel(values, reference, use_reference=True)
    assert referenced[0, 0] == 0.0
    assert referenced[1, 0] == 0.5
    assert referenced[1, 2] == 1.0
    assert referenced[2, 2] == 1.0

    ignored = normalize_display_channel(values, reference, use_reference=False)
    assert ignored[1, 1] == 0.5
    assert ignored[1, 2] == 0.625


def test_hyperspectral_image_export_uses_display_orientation():
    np.testing.assert_array_equal(
        orient_image_for_export(np.array([[1, 2], [3, 4]])),
        np.array([[3, 4], [1, 2]]),
    )
    rgb_rows = np.array([[[1, 2, 3]], [[4, 5, 6]]])
    np.testing.assert_array_equal(
        orient_image_for_export(rgb_rows),
        np.array([[[4, 5, 6]], [[1, 2, 3]]]),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "display.png"
        rgb = np.array([
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ])
        save_display_image(str(path), rgb)
        from matplotlib import image as matplotlib_image

        saved = matplotlib_image.imread(path)
        np.testing.assert_allclose(saved[0, 0, :3], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(saved[1, 0, :3], [1.0, 0.0, 0.0])

        try:
            save_display_image(str(Path(directory) / "display.svg"), rgb)
        except ValueError as error:
            assert "PNG or JPEG" in str(error)
        else:
            raise AssertionError("Unsupported display-image extension accepted")


def test_spectral_planes_put_maximum_z_away_from_xy_quadrant():
    data = np.fromfunction(lambda y, x, z: 100*y + 10*x + z, (2, 3, 4))
    sections = hyperspectral_sections(data, ix=1, iy=0, iz=0)
    descending_z = np.array([700.0, 600.0, 500.0, 400.0])
    xz, yz, limits = orient_spectral_planes(sections, descending_z)

    assert limits == (400.0, 700.0)
    # The maximum Z is the last display row (top) and column (right).
    np.testing.assert_array_equal(xz[-1], sections["xz"][0])
    np.testing.assert_array_equal(yz[:, -1], sections["yz"][:, 0])


def test_z_slice_colormap_inversion():
    assert display_colormap("viridis", False) == "viridis"
    assert display_colormap("viridis", True) == "viridis_r"
    assert display_colormap("viridis_r", True) == "viridis"


def test_every_registered_colormap_can_be_selected_and_inverted():
    names = tuple(sorted(colormaps))
    assert "viridis" in names
    assert "twilight" in names  # Matplotlib's cyclic map is available.
    for name in names:
        assert display_colormap(name, False) == name
        assert display_colormap(name, True) in names


def test_hyperspectral_colormaps_follow_documented_groups_without_reverses():
    groups = grouped_colormaps()
    assert [category for category, _ in groups] == [
        "Perceptually Uniform Sequential",
        "Sequential",
        "Sequential (2)",
        "Diverging",
        "Cyclic",
        "Qualitative",
        "Miscellaneous",
    ]
    grouped_names = [name for _, names in groups for name in names]
    assert grouped_names[:5] == [
        "viridis", "plasma", "inferno", "magma", "cividis"
    ]
    assert not any(name.endswith("_r") for name in grouped_names)
    assert set(grouped_names) == {
        name for name in colormaps if not name.endswith("_r")
    }


def test_export_file_types_and_raster_shapes():
    assert {pattern for _, pattern in PLOT_EXPORT_FILETYPES} == {
        "*.png", "*.jpg *.jpeg", "*.svg", "*.pdf", "*.eps"
    }
    assert {pattern for _, pattern in data_export_filetypes("image")} == {
        "*.csv", "*.txt", "*.png", "*.jpg *.jpeg"
    }
    assert {pattern for _, pattern in data_export_filetypes("curve")} == {
        "*.csv", "*.txt"
    }
    assert {pattern for _, pattern in data_export_filetypes("hyperspectral")} == {
        "*.csv", "*.txt"
    }
    image = DataSet(
        "image", np.arange(6.0).reshape(2, 3),
        [np.arange(3.0), np.arange(2.0)], "Image", "source.txt"
    )
    assert raster_export_array(image).shape == (2, 3)


def test_common_curve_text_formats_import():
    cases = {
        "curve.csv": (
            "Wavelength,Signal\n500,10\n510,12\n520,9\n",
            np.array([500.0, 510.0, 520.0]),
            np.array([10.0, 12.0, 9.0]),
            "two_column",
        ),
        "curve.txt": (
            "# Instrument = Test\n500\t10\n510\t12\n520\t9\n",
            np.array([500.0, 510.0, 520.0]),
            np.array([10.0, 12.0, 9.0]),
            "two_column",
        ),
        "single_column.txt": (
            "10\n12\n9\n",
            np.arange(3.0),
            np.array([10.0, 12.0, 9.0]),
            "one_column",
        ),
    }
    with tempfile.TemporaryDirectory() as directory:
        for filename, (content, expected_x, expected_y, parser) in cases.items():
            path = Path(directory) / filename
            path.write_text(content)
            detection = detect_text(str(path))[0]
            assert detection.kind == "curve"
            assert detection.parser == parser
            loaded = load_text(str(path), detection)
            assert loaded.kind == "curve"
            np.testing.assert_allclose(loaded.axes[0], expected_x)
            np.testing.assert_allclose(loaded.data, expected_y)


def test_raman_y_values_import_as_curve_with_index_axis():
    source_path = SAMPLES / "ramanSpecRshift.txt"
    values_path = SAMPLES / "ramanSpecRshiftY.txt"

    source = load_text(str(source_path), detect_text(str(source_path))[0])
    detection = detect_text(str(values_path))[0]
    loaded = load_text(str(values_path), detection)

    assert detection.kind == "curve"
    assert detection.parser == "one_column"
    assert loaded.kind == "curve"
    np.testing.assert_array_equal(loaded.axes[0], np.arange(source.data.size))
    np.testing.assert_array_equal(loaded.data, source.data)


def test_jupyter_array_and_file_import_api():
    for shape, kind, axis_sizes in (
        ((7,), "curve", (7,)),
        ((3, 5), "image", (5, 3)),
        ((3, 5, 7), "hyperspectral", (5, 3, 7)),
    ):
        dataset = dataset_from_array(np.arange(np.prod(shape)).reshape(shape))
        assert dataset.kind == kind
        assert tuple(map(len, dataset.axes)) == axis_sizes
    dataset = notebook_import_data(str(SAMPLES / "ramanSpecRshiftY.txt"))
    data, axes, metadata = import_numpy(str(SAMPLES / "ramanSpecRshiftY.txt"))
    assert dataset.kind == "curve"
    np.testing.assert_array_equal(data, dataset.data)
    np.testing.assert_array_equal(axes[0], np.arange(len(data)))
    assert metadata == dataset.metadata


def test_jupyter_extraction_and_numpy_persistence():
    dataset = dataset_from_array(
        np.arange(12.0).reshape(3, 4),
        title="Notebook image", metadata={"instrument": "test"},
    )
    view = NotebookView(dataset)
    extracted = view.extract()
    np.testing.assert_array_equal(extracted["data"], dataset.data)
    assert extracted["metadata"]["kind"] == "image"
    assert extracted["metadata"]["metadata"] == {"instrument": "test"}
    assert view.to_scipy().shape == (3, 4)
    with tempfile.TemporaryDirectory() as directory:
        npy = Path(directory) / "data.npy"
        npz = Path(directory) / "data.npz"
        mat = Path(directory) / "data.mat"
        view.save_numpy(str(npy)); view.save_numpy(str(npz)); view.save_scipy(str(mat))
        np.testing.assert_array_equal(np.load(npy), dataset.data)
        with np.load(npz) as saved:
            np.testing.assert_array_equal(saved["data"], dataset.data)
            np.testing.assert_array_equal(saved["axis_0"], dataset.axes[0])
            np.testing.assert_array_equal(saved["axis_1"], dataset.axes[1])
        from scipy.io import loadmat
        np.testing.assert_array_equal(loadmat(mat)["data"], dataset.data)


def test_jupyter_controls_and_optional_dependencies_are_documented():
    source = (ROOT / "hyperscope.py").read_text()
    for label in (
        "Cursor 1", "Cursor 2", "Between Markers", "Data color",
        "Fit color", "Show Contrast Reference", "Select Contrast Box",
        "RGB Channels", "Z Slice", "Zero", "Apply Axes",
    ):
        assert label in source
    assert "ipywidgets" in source and "ipympl" in source


def test_notebook_sliders_display_physical_coordinates_and_keep_indices():
    options = coordinate_slider_options(
        np.array([500.0, 512.5, 530.0]), 'cm⁻¹'
    )
    assert options == [
        ('500 cm⁻¹', 0), ('512.5 cm⁻¹', 1), ('530 cm⁻¹', 2)
    ]
    source = (ROOT / "hyperscope.py").read_text()
    curve_source = source.split("class CurveNotebookView", 1)[1].split(
        "class ImageNotebookView", 1
    )[0]
    image_source = source.split("class ImageNotebookView", 1)[1].split(
        "class HyperspectralNotebookView", 1
    )[0]
    hyperspectral_source = source.split(
        "class HyperspectralNotebookView", 1
    )[1]
    assert "w.SelectionRangeSlider" in curve_source
    assert image_source.count("w.SelectionSlider") >= 4
    assert hyperspectral_source.count("w.SelectionSlider") >= 3
    assert "w.SelectionRangeSlider" in hyperspectral_source
    assert "w.IntSlider" not in source
    assert "w.IntRangeSlider" not in source


def test_hyperspectral_notebook_slider_callback_updates_position_and_draws():
    class Slider:
        def __init__(self, value):
            self.value = value

    view = object.__new__(HyperspectralNotebookView)
    view.x_slider, view.y_slider, view.z_slider = (
        Slider(4), Slider(3), Slider(8)
    )
    view._updating_sliders = False
    calls = []
    view._draw = lambda: calls.append("draw")

    view._coordinates_changed()

    assert view.pos == [4, 3, 8]
    assert calls == ["draw"]
    calls.clear()
    view._set_position(ix=6, iy=5, iz=9)
    assert view.pos == [6, 5, 9]
    assert [view.x_slider.value, view.y_slider.value, view.z_slider.value] == [
        6, 5, 9
    ]
    assert calls == ["draw"]
    source = (ROOT / "hyperscope.py").read_text()
    hyperspectral_source = source.split(
        "class HyperspectralNotebookView", 1
    )[1]
    assert "slider.observe(self._coordinates_changed" in hyperspectral_source
    assert "self.figure.canvas.draw_idle()" in hyperspectral_source


def test_hyperspectral_notebook_has_profiles_contrast_and_spectral_dragging():
    source = (ROOT / "hyperscope.py").read_text()
    hyperspectral_source = source.split(
        "class HyperspectralNotebookView", 1
    )[1]
    for name in (
        "self.ax_x_profile", "self.ax_y_profile",
        "RectangleSelector", "self._contrast_selected",
        "self.ax_spectrum.axvspan", "'z-spectrum'", "'z-xz'", "'z-yz'",
    ):
        assert name in hyperspectral_source
    assert "self.ax_x_profile.plot(x, s['x_profile']" in hyperspectral_source
    assert "self.ax_y_profile.plot(s['y_profile'], y" in hyperspectral_source
    assert "self._updating_sliders = True" in hyperspectral_source


def test_image_notebook_views_display_one_widget_canvas_with_controls():
    class Widget:
        pass

    class VBox(Widget):
        def __init__(self, children):
            self.children = children

    Widgets = type("Widgets", (), {"Widget": Widget, "VBox": VBox})

    canvas = Widget()
    controls = Widget()
    view = type("View", (), {})()
    view.figure = type("Figure", (), {"canvas": canvas})()
    view.controls = controls
    displayed = []
    original_widgets = notebook_api._widgets
    notebook_api._widgets = lambda: (Widgets, displayed.append)
    try:
        returned = notebook_api._show_widget_canvas(view)
    finally:
        notebook_api._widgets = original_widgets

    assert returned is view
    assert displayed == [view._display_widget]
    assert view._display_widget.children == [canvas, controls]
    source = (ROOT / "hyperscope.py").read_text()
    image_source = source.split("class ImageNotebookView", 1)[1].split(
        "class HyperspectralNotebookView", 1
    )[0]
    hyperspectral_source = source.split(
        "class HyperspectralNotebookView", 1
    )[1]
    assert "with plt.ioff():" in image_source
    assert "with plt.ioff():" in hyperspectral_source
    assert "return _show_widget_canvas(self)" in image_source
    assert "return _show_widget_canvas(self)" in hyperspectral_source


def test_curve_notebook_click_moves_cursor_and_uses_live_canvas():
    class Value:
        def __init__(self, value):
            self.value = value

    view = object.__new__(CurveNotebookView)
    view.dataset = DataSet(
        "curve", np.arange(10.0), [np.arange(10.0)], "Curve", "notebook"
    )
    view.axis = object()
    view.show_c1, view.show_c2 = Value(True), Value(True)
    view.cursors = [2, 8]
    view.cursor_slider = Value(tuple(view.cursors))
    view._updating_cursor_slider = False
    view.drag_cursor = None
    draws = []
    view._draw = lambda: draws.append("draw")
    event = type("Event", (), {"inaxes": view.axis, "xdata": 6.1})()

    view._press(event)

    assert view.drag_cursor == 1
    assert view.cursors == [2, 6]
    assert view.cursor_slider.value == (2, 6)
    assert draws == ["draw"]
    source = (ROOT / "hyperscope.py").read_text()
    curve_source = source.split("class CurveNotebookView", 1)[1].split(
        "class ImageNotebookView", 1
    )[0]
    assert "with plt.ioff():" in curve_source
    assert "return _show_widget_canvas(self)" in curve_source
    assert "button_press_event" in curve_source


def test_curve_notebook_fit_values_are_exposed_and_displayed():
    view = object.__new__(CurveNotebookView)
    view.fit = np.array([1.0, 2.0, 3.0])
    values = view.fit_values()
    np.testing.assert_array_equal(values, view.fit)
    values[0] = 99.0
    assert view.fit[0] == 1.0

    source = (ROOT / "hyperscope.py").read_text()
    curve_source = source.split("class CurveNotebookView", 1)[1].split(
        "class ImageNotebookView", 1
    )[0]
    assert "self.fit_output = w.Output" in curve_source
    assert "print('Fit parameters:')" in curve_source
    assert "print('Fit values:')" not in curve_source
    assert "threshold=self.fit.size" not in curve_source


def test_2d_notebook_draw_uses_only_its_four_axes():
    source = (ROOT / "hyperscope.py").read_text()
    image_source = source.split("class ImageNotebookView", 1)[1].split(
        "class HyperspectralNotebookView", 1
    )[0]
    assert "for ax in self.axes.ravel():" in image_source
    for hyperspectral_axis in (
        "ax_x_profile", "ax_xz", "ax_spectrum", "ax_yz", "ax_y_profile"
    ):
        assert hyperspectral_axis not in image_source
    for feature in (
        "Show Second Cursor", "self.x2_slider", "self.y2_slider",
        "image_line_profile(self.dataset, self.pos, self.pos2)",
        "'second_cursor': list(self.pos2)",
    ):
        assert feature in image_source


def test_hyperspectral_notebook_rgb_range_reducer_dropdown():
    source = (ROOT / "hyperscope.py").read_text()
    hyperspectral_source = source.split(
        "class HyperspectralNotebookView", 1
    )[1]
    assert "options=['Average', 'Maximum', 'Minimum']" in hyperspectral_source
    assert "self.ignore_reference, self.combine" in hyperspectral_source
    assert "self.combine.value" in hyperspectral_source
    assert "'range_value': self.combine.value" in hyperspectral_source


def test_imported_curve_style_renders_with_matplotlib():
    from matplotlib.figure import Figure

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "curve.csv"
        path.write_text("Wavelength,Signal\n500,10\n510,12\n520,9\n")
        dataset = load_text(str(path), detect_text(str(path))[0])
        figure = Figure()
        axis = figure.add_subplot(111)
        lines = axis.plot(
            dataset.axes[0], dataset.data, **default_curve_style()
        )
        assert len(lines) == 1
        assert lines[0].get_linewidth() == 1.5


def test_inline_curve_display_and_fit_controls():
    style = curve_style_from_controls(
        "--", "2", "navy", "o", "8", "orange"
    )
    assert style == {
        "linestyle": "--",
        "linewidth": 2.0,
        "color": "navy",
        "marker": "o",
        "markersize": 8.0,
        "markerfacecolor": "orange",
        "markeredgecolor": "orange",
    }
    x = np.arange(5.0)
    y = 2 * x + 1
    fitted, parameters = fit_curve_selection(
        x, y, "line", "", True, [1, 3]
    )
    np.testing.assert_allclose(fitted, y)
    np.testing.assert_allclose(parameters, [2.0, 1.0], atol=1e-8)
    polynomial, _ = fit_curve_selection(
        x, y, "polynomial", "order=2", False, [1, 3]
    )
    np.testing.assert_allclose(polynomial, y, atol=1e-8)
    fit_style = default_fit_style()
    assert fit_style["color"] == "orange"
    assert fit_style["linewidth"] == 2.0
    line_parameters = labeled_fit_parameters("line", parameters)
    assert list(line_parameters) == ["Slope", "Intercept"]
    np.testing.assert_allclose(list(line_parameters.values()), [2.0, 1.0])
    assert list(
        labeled_fit_parameters(
            "gaussian", np.array([8.0, 500.0, 2.0, 1.0])
        )
    ) == ["Amplitude", "Center", "Sigma", "Baseline"]
    assert list(
        labeled_fit_parameters(
            "polynomial", np.array([1.0, 2.0, 3.0, 4.0])
        )
    ) == [
        "X^3 Coefficient", "X^2 Coefficient",
        "Linear Coefficient", "Intercept",
    ]
    visible_fit = fit_sidebar_information(
        True, "line", fitted, line_parameters
    )
    assert list(visible_fit) == [
        "Fit", "Fit Function", "Fit Parameters", "Slope", "Intercept"
    ]
    assert fit_sidebar_information(
        False, "line", fitted, line_parameters
    ) == {}
    sidebar = metadata_sidebar_text(
        DataSet(
            "curve", y, [x], "Curve", "curve.csv",
            units=["x", "y"], labels=["X", "Y"],
        ),
        visible_fit,
    )
    assert "\n\nFit Parameters\n" in sidebar
    assert curve_axis_scale(np.array([1.0, 10.0]), True) == "log"
    assert curve_axis_scale(np.array([-1.0, 10.0]), False) == "linear"
    try:
        curve_axis_scale(np.array([0.0, 10.0]), True)
    except ValueError:
        pass
    else:
        raise AssertionError("A non-positive logarithmic axis was accepted")


def test_curve_cursors_have_distinct_colors_and_range_statistics():
    assert CURVE_CURSOR_COLORS == ("red", "green")
    statistics = curve_cursor_statistics(
        np.array([0.0, 3.0, 6.0]),
        np.array([1.0, 5.0, 2.0]),
        [2, 0],
    )
    assert statistics == {
        "X Difference": -6.0,
        "Y Difference": -1.0,
        "Distance": np.sqrt(37.0),
        "Average": 8.0 / 3.0,
        "Minimum": 1.0,
        "Maximum": 5.0,
    }
    whole_curve = curve_cursor_statistics(
        np.array([0.0, 3.0, 6.0]),
        np.array([1.0, 5.0, 2.0]),
        [2, 0],
        cursors_visible=False,
    )
    assert whole_curve == {
        "X Range": 6.0,
        "Y Range": 4.0,
        "Average": 8.0 / 3.0,
        "Minimum": 1.0,
        "Maximum": 5.0,
    }
    assert "Distance" not in whole_curve

    one_cursor = curve_sidebar_information(
        np.array([0.0, 3.0, 6.0]),
        np.array([1.0, 5.0, 2.0]),
        [2, 0],
        [True, False],
    )
    assert one_cursor["Cursor 1 (red)"] == (6.0, 2.0)
    assert all(type(value) is float for value in one_cursor["Cursor 1 (red)"])
    assert "Cursor 2 (green)" not in one_cursor
    assert "Across Data" in one_cursor
    assert "X Range" in one_cursor and "Y Range" in one_cursor
    assert "Distance" not in one_cursor

    two_cursors = curve_sidebar_information(
        np.array([0.0, 3.0, 6.0]),
        np.array([1.0, 5.0, 2.0]),
        [2, 0],
        [True, True],
    )
    assert "Between Cursors" in two_cursors
    assert "X Difference" in two_cursors
    assert "Y Difference" in two_cursors
    assert "Distance" in two_cursors


def test_curve_cursor_range_creates_inclusive_child_view():
    source = DataSet(
        "curve",
        np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        [np.array([100.0, 110.0, 120.0, 130.0, 140.0])],
        "Parent curve",
        "curve.csv",
        {},
        ["nm", "counts"],
        ["Wavelength", "Signal"],
    )
    child = curve_range_dataset(source, [3, 1])
    assert child.kind == "curve"
    np.testing.assert_array_equal(child.data, [20.0, 30.0, 40.0])
    np.testing.assert_array_equal(child.axes[0], [110.0, 120.0, 130.0])
    assert child.parent == source.uid
    assert child.derivation == {
        "start_index": 1,
        "end_index": 3,
        "start_x": 110.0,
        "end_x": 130.0,
    }


def test_duplicate_and_axis_replacement_create_child_views():
    source = DataSet(
        "curve", np.arange(4.0), [np.linspace(500, 530, 4)],
        "Axis source", "source.csv", {}, ["nm", "counts"],
        ["Wavelength", "Signal"],
    )
    duplicate = duplicate_dataset(source)
    assert duplicate.parent == source.uid
    assert duplicate.uid != source.uid
    assert duplicate.derivation == {"operation": "duplicate"}
    assert not np.shares_memory(duplicate.data, source.data)

    target_curve = DataSet(
        "curve", np.arange(4.0) + 10, [np.arange(4.0)],
        "Target curve", "target.csv", {}, ["index", "a.u."],
        ["Index", "Signal"],
    )
    curve_child = dataset_with_replaced_axis(source, target_curve)
    assert curve_child.parent == target_curve.uid
    np.testing.assert_array_equal(curve_child.axes[0], source.axes[0])
    assert curve_child.labels[0] == "Wavelength"
    assert curve_child.units[0] == "nm"

    target_hyperspectral = DataSet(
        "hyperspectral", np.arange(24.0).reshape(2, 3, 4),
        [np.arange(3.0), np.arange(2.0), np.arange(4.0)],
        "Target hyperspectral", "target.txt", {},
        ["mm", "mm", "index"], ["X", "Y", "Z"],
    )
    hyper_child = dataset_with_replaced_axis(source, target_hyperspectral)
    assert hyper_child.parent == target_hyperspectral.uid
    np.testing.assert_array_equal(hyper_child.axes[2], source.axes[0])
    assert hyper_child.labels[2] == "Wavelength"
    assert hyper_child.derivation["target_axis"] == "Z"

    target_image = DataSet(
        "image", np.zeros((2, 2)), [np.arange(2), np.arange(2)],
        "Image", "image.txt",
    )
    try:
        dataset_with_replaced_axis(source, target_image)
    except ValueError:
        pass
    else:
        raise AssertionError("A Curve axis was sent to a 2D image")


def test_hyperspectral_range_value_methods():
    chunk = np.array([
        [[1.0, 4.0, 7.0], [2.0, 8.0, 5.0]],
        [[3.0, 6.0, 9.0], [0.0, 3.0, 6.0]],
    ])
    np.testing.assert_array_equal(
        collapse_spectral_range(chunk, "Average"),
        np.array([[4.0, 5.0], [6.0, 3.0]])
    )
    np.testing.assert_array_equal(
        collapse_spectral_range(chunk, "Maximum"),
        np.array([[7.0, 8.0], [9.0, 6.0]])
    )
    np.testing.assert_array_equal(
        collapse_spectral_range(chunk, "Minimum"),
        np.array([[1.0, 2.0], [3.0, 0.0]])
    )

def test_2d_contrast_reference_limits():
    image = np.arange(20.0).reshape(4, 5)
    assert image_contrast_limits(image, [[1, 1], [3, 2]]) == (6.0, 13.0)

def test_hyperspectral_slice_to_new_view():
    source = DataSet(
        "hyperspectral",
        np.arange(60.0).reshape(3, 4, 5),
        [np.arange(4.0), np.arange(3.0), np.linspace(500, 540, 5)],
        "Parent", "source.txt", {},
        ["um", "um", "nm"], ["X", "Y", "Wavelength"]
    )
    derived = hyperspectral_slice_dataset(source, 3)
    assert derived.kind == "image"
    np.testing.assert_array_equal(derived.data, source.data[:, :, 3])
    np.testing.assert_array_equal(derived.axes[0], source.axes[0])
    np.testing.assert_array_equal(derived.axes[1], source.axes[1])
    assert derived.parent == source.uid
    assert derived.derivation == {"z_index": 3, "z_value": 530.0}


def test_image_cursor_profiles_create_curve_children():
    source = DataSet(
        "image",
        np.arange(20.0).reshape(4, 5),
        [np.linspace(10, 50, 5), np.linspace(-3, 0, 4)],
        "Map", "map.txt", {},
        ["µm", "µm", "counts"], ["X", "Y", "Intensity"],
    )
    x_profile = image_profile_dataset(source, "x", ix=3, iy=2)
    y_profile = image_profile_dataset(source, "y", ix=3, iy=2)

    assert x_profile.kind == y_profile.kind == "curve"
    np.testing.assert_array_equal(x_profile.data, source.data[2, :])
    np.testing.assert_array_equal(x_profile.axes[0], source.axes[0])
    np.testing.assert_array_equal(y_profile.data, source.data[:, 3])
    np.testing.assert_array_equal(y_profile.axes[0], source.axes[1])
    assert x_profile.labels == ["X", "Intensity"]
    assert y_profile.labels == ["Y", "Intensity"]
    assert x_profile.units == ["µm", "counts"]
    assert y_profile.units == ["µm", "counts"]
    assert x_profile.parent == y_profile.parent == source.uid
    assert x_profile.derivation == {"axis": "x", "y_index": 2}
    assert y_profile.derivation == {"axis": "y", "x_index": 3}


def test_image_line_profile_uses_physical_distance():
    source = DataSet(
        "image",
        np.arange(20.0).reshape(4, 5),
        [np.arange(5.0) * 2.0, np.arange(4.0) * 3.0],
        "Map", "map.txt",
    )
    distance, values = image_line_profile(source, (1, 1), (4, 3))

    assert len(distance) == len(values) == 4
    assert distance[0] == 0.0
    assert np.isclose(distance[-1], np.hypot(6.0, 6.0))
    assert values[0] == source.data[1, 1]
    assert values[-1] == source.data[3, 4]


def test_tool_instructions_are_shown_only_once():
    seen = set()
    assert consume_first_use(seen, "contrast_reference")
    assert not consume_first_use(seen, "contrast_reference")
    assert consume_first_use(seen, "channel_range")
    assert not consume_first_use(seen, "channel_range")

def test_profile_cursor_coordinate_mapping():
    ascending = np.array([100.0, 110.0, 125.0, 150.0])
    descending = ascending[::-1]
    assert nearest_axis_index(ascending, 121.0) == 2
    assert nearest_axis_index(descending, 121.0) == 1


def test_profile_cursor_visibility_keeps_curve_cursor_active():
    hidden_auxiliary = profile_cursor_visibility(False)
    assert hidden_auxiliary == {"curve": True, "xz": False, "yz": False}
    assert profile_cursor_visibility(True) == {
        "curve": True, "xz": True, "yz": True
    }


def test_hyperspectral_initial_position_uses_rightmost_z_value():
    ascending = np.linspace(500, 600, 11)
    descending = ascending[::-1]
    assert initial_hyperspectral_position((5, 7, 11), ascending) == [3, 2, 10]
    assert initial_hyperspectral_position((5, 7, 11), descending) == [3, 2, 0]


def test_file_metadata_follows_view_information():
    dataset = DataSet(
        "image", np.zeros((2, 3)), [np.arange(3), np.arange(2)],
        "Image", "image.txt", {"Operator": "Ada"}, ["mm", "mm", "counts"],
        ["X", "Y", "Value"],
    )
    text = metadata_sidebar_text(
        dataset, {"View type": "2D image", "Cursor X": 1.0}
    )
    assert text.index("View type:") < text.index("Cursor X:")
    assert text.index("Cursor X:") < text.index("Metadata:")
    assert text.index("Metadata:") < text.index("Operator:")
    hidden = metadata_sidebar_text(
        dataset, {"View type": "2D image", "Cursor X": 1.0}, False
    )
    assert "View type: 2D image" in hidden
    assert "Cursor X: 1.0" in hidden
    assert "Metadata:" not in hidden
    assert "Operator:" not in hidden


def test_data_export_import_roundtrip():
    datasets = [
        DataSet(
            "curve", np.linspace(2, 8, 5), [np.linspace(500, 540, 5)],
            "Curve title", "source.dat", {}, ["nm", "counts"],
            ["Wavelength", "Signal"],
        ),
        DataSet(
            "image", np.arange(12.0).reshape(3, 4),
            [np.linspace(-1, 2, 4), np.linspace(10, 30, 3)],
            "Image title", "source.dat", {}, ["um", "um", "a.u."],
            ["X", "Y", "Value"],
        ),
        DataSet(
            "hyperspectral", np.arange(24.0).reshape(2, 3, 4),
            [np.linspace(1, 3, 3), np.linspace(5, 6, 2), np.linspace(600, 630, 4)],
            "Hyperspectral title", "source.dat", {}, ["mm", "mm", "nm"],
            ["X", "Y", "Wavelength"],
        ),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for dataset in datasets:
            for suffix in (".csv", ".txt"):
                path = Path(directory) / f"{dataset.kind}{suffix}"
                export_dataset_text(dataset, str(path))
                detection = detect_text(str(path))[0]
                assert detection.kind == dataset.kind
                assert detection.parser == "hyperscope_export"
                loaded = load_text(str(path), detection)
                assert loaded.kind == dataset.kind
                assert loaded.title == dataset.title
                assert loaded.labels == dataset.labels
                assert loaded.units == dataset.units
                np.testing.assert_allclose(loaded.data, dataset.data)
                for actual, expected in zip(loaded.axes, dataset.axes):
                    np.testing.assert_allclose(actual, expected)


def test_plain_matrix_csv_imports_as_2d_image():
    matrix = np.arange(30.0).reshape(5, 6)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "third_party_image.csv"
        np.savetxt(path, matrix, delimiter=",")
        detection = detect_text(str(path))[0]
        assert detection.kind == "image"
        assert detection.parser == "matrix"
        loaded = load_text(str(path), detection)
        np.testing.assert_array_equal(loaded.data, matrix)


def test_requested_user_facing_terms():
    source = (
        (ROOT / "hyperscope_desktop.py").read_text()
        + (ROOT / "hyperscope.py").read_text()
    )
    assert 'text="Open Views"' in source
    assert 'text="Contrast Reference"' in source
    assert 'text="Show Contrast Reference"' in source
    assert 'text="Hide Contrast Reference"' not in source
    assert 'text="Ignore Contrast Reference"' in source
    assert 'text="Invert Colors"' in source
    assert "self.title_row.pack(fill=\"x\")" in source
    assert "self.new_view_bar.pack(fill=\"x\")" in source
    assert "self.export_bar.pack(fill=\"x\")" in source
    assert source.index("self.title_row.pack") < source.index("self.new_view_bar.pack")
    assert source.index("self.new_view_bar.pack") < source.index("self.export_bar.pack")
    assert 'self.new_view_bar, text="Line Profile to New View"' in source
    assert "self.new_view_bar, text=text, command=command" in source
    assert 'text="To New View:"' in source
    assert 'text="Export:"' in source
    assert 'text="Export Plot to File"' in source
    assert source.count('text="Export Image to File"') == 2
    assert 'text="Export Data to File"' in source
    assert 'text="Metadata", command=' not in source
    assert 'self.bottom,text="Export line"' not in source
    assert 'ttk.Button(self.bottom, text=text, command=command)' not in source
    assert "NavigationToolbar2Tk" not in source
    assert 'text="Show Metadata"' in source
    assert 'text="Show Sidebar"' not in source
    assert "self.body.forget" not in source
    assert 'text="Show Cursors"' in source
    assert 'text=f"Show Cursor {index + 1} ({color})"' in source
    assert 'text="Show Vertical Cursors"' in source
    assert 'text="Show Second Cursor"' in source
    assert 'text="Show Profile Cursors"' in source
    assert "self.view_controls.pack(fill=\"x\")" in source
    assert source.index("self.view_controls.pack") < source.index("self.bottom.pack")
    assert 'text="Sidebar", command=self.toggle_sidebar' not in source
    assert 'text="Cursors", command=self.toggle_cursors' not in source
    assert 'values=["Z Slice", "RGB Channels"]' in source
    assert 'text="Value:"' in source
    assert 'values=["Average", "Maximum", "Minimum"]' in source
    assert '"*.txt *.csv *.hscope"' in source
    assert "Peak Height" not in source
    for label in (
        "Spectrum to New View",
        "X Profile to New View",
        "Y Profile to New View",
        "2D Slice to New View",
    ):
        assert label in source
    assert '"curve": "Curve"' in source
    assert 'text="Range Within Cursors to New View"' in source
    assert 'marker="o"' in source
    assert '"Between Cursors"' in source
    assert '"Across Data"' in source
    assert 'sharex=self.ax_xy' in source
    assert 'sharey=self.ax_xy' in source
    image_code = source.split("class ImageView", 1)[1].split(
        "class HyperspectralView", 1
    )[0]
    curve_code = source.split("class CurveView", 1)[1].split(
        "class ImageView", 1
    )[0]
    hyperspectral_code = source.split("class HyperspectralView", 1)[1].split(
        "class ImportDialog", 1
    )[0]
    assert ".set_title(" not in image_code
    assert ".set_title(" not in hyperspectral_code
    assert "self.fig.suptitle(self.ds.title)" in image_code
    assert "self.fig.suptitle(self.ds.title)" in hyperspectral_code
    assert 'text="Send X Axis to Other View"' in curve_code
    assert 'text="Send X"' not in curve_code
    for label in (
        "Parameters:",
        "Between Markers", "Apply Axes / Labels", "X Ticks…", "Y Ticks…",
    ):
        assert f'text="{label}"' in curve_code
    for label in (
        "Line style", "Line width", "Line color",
        "Marker", "Marker size", "Marker color",
    ):
        assert f'("{label}",' in curve_code
    assert 'self.fit_controls, text="Fit"' in curve_code
    assert "self.axis_role_names = [\"Original X\", \"Original Y\"]" in curve_code
    assert 'text="Show Fit"' in curve_code
    assert 'text=f"Log {role[0]}"' in curve_code
    assert 'text="Fit Style:"' in curve_code
    assert '"Fit Parameters"' in source
    assert "fit_sidebar_information(" in curve_code
    assert "self.show_fit_var.get()" in curve_code
    assert "configure(height=420)" in curve_code
    assert 'text="Duplicate View"' in source
    assert 'text="Delete View"' in source
    assert "dataset_with_replaced_axis(" in curve_code
    assert 'view.ds.kind in {"curve", "hyperspectral"}' in curve_code
    assert 'zorder=10' in curve_code
    assert curve_code.index("self.ax.plot(x, y") < curve_code.index(
        "x, self.fit_line"
    )
    assert curve_code.index('text="Data:"') < curve_code.index(
        "self.fit_style_controls.pack"
    )
    assert curve_code.index("self.fit_style_controls.pack") < curve_code.index(
        "self.fit_controls.pack"
    )
    assert curve_code.index("self.fit_controls.pack") < curve_code.index(
        "self.axis_controls.pack"
    )
    assert image_code.index('text="Show Contrast Reference"') < image_code.index(
        'text="Ignore Contrast Reference"'
    )
    assert hyperspectral_code.index(
        'text="Show Contrast Reference"'
    ) < hyperspectral_code.index('text="Ignore Contrast Reference"')
    assert (
        'self.view_controls, text="Show Contrast Reference"' in image_code
    )
    assert (
        'self.view_controls, text="Show Contrast Reference"'
        in hyperspectral_code
    )
    assert "grouped_colormap_button(" in image_code
    assert "grouped_colormap_button(" in hyperspectral_code
    assert '"X Profile to New View"' in image_code
    assert '"Y Profile to New View"' in image_code
    assert "image_profile_dataset(" in image_code
    assert 'text="Export Image to File"' in image_code
    assert "save_display_image(" in image_code
    assert 'text="Export Image to File"' in hyperspectral_code
    assert "Export 2D Slice to File" not in hyperspectral_code
    assert "def current_display_image" in hyperspectral_code
    assert 'self.mode.get() == "RGB Channels"' in hyperspectral_code
    assert "save_display_image(" in hyperspectral_code
    assert "ExportNormalizationDialog" not in source
    assert "prepare_export_image" not in source
    assert "before=self.export_data_button" in hyperspectral_code
    assert '"Spectrum to New View"' in hyperspectral_code
    assert "values=available_colormaps()" not in hyperspectral_code
    assert 'self.tree.bind("<<TreeviewSelect>>", self.select_tree)' in source
    assert 'self.tree.bind("<Double-1>"' not in source
    assert 'self.drag = "z-xz"' in hyperspectral_code
    assert 'self.drag = "z-yz"' in hyperspectral_code
    assert "orient_spectral_planes(sections, z)" in hyperspectral_code
    assert 'cursor_visibility["curve"]' in hyperspectral_code
    assert 'self.cmap.trace_add("write", lambda *_: self.draw())' in image_code
    assert 'self.ax_x.tick_params(axis="x", labelbottom=False)' in image_code
    assert 'self.ax_y.tick_params(axis="y", labelleft=False)' in image_code
    assert 'self.ax_xz.tick_params(axis="x", labelbottom=False)' in hyperspectral_code
    assert 'self.ax_yz.tick_params(axis="y", labelleft=False)' in hyperspectral_code
    assert 'text="Open windows"' not in source
    assert 'text="Reference box"' not in source
    assert '"line": "row_line"' not in source
    assert 'text="Curve row index (for row-based files):"' in source

def test_session_schema_is_strict():
    valid = {"version": APP_VERSION, "datasets": {}, "states": {}}
    validate_session_payload(valid)
    for invalid in (
        {"version": APP_VERSION - 1, "datasets": {}, "states": {}},
        {"version": APP_VERSION, "datasets": [], "states": {}},
        {"version": APP_VERSION, "datasets": {}, "states": []},
    ):
        try:
            validate_session_payload(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid session accepted: {invalid}")
