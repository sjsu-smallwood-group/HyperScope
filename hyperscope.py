#!/usr/bin/env python3
"""HyperScope data, analysis, import, export, and Jupyter view API.

The module is Tk-free. Interactive notebook controls import ipywidgets lazily.
"""
from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage, optimize, special

from matplotlib import colormaps
from matplotlib.colors import CSS4_COLORS
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

APP_VERSION = 18
PLOT_EXPORT_FILETYPES = [
    ("PNG image", "*.png"),
    ("JPEG image", "*.jpg *.jpeg"),
    ("SVG vector image", "*.svg"),
    ("PDF document", "*.pdf"),
    ("EPS vector image", "*.eps"),
]
CURVE_CURSOR_COLORS = ("red", "green")
CURVE_LINE_STYLES = ("-", "--", "-.", ":", "None")
CURVE_LINE_WIDTHS = ("0.5", "1", "1.5", "2", "3", "4", "5")
CURVE_MARKERS = ("None", ".", "o", "s", "^", "v", "D", "x", "+", "*")
CURVE_MARKER_SIZES = ("2", "4", "6", "8", "10", "12", "16")
CURVE_COLORS = ("#1565c0", *tuple(sorted(CSS4_COLORS)))
CURVE_FIT_FUNCTIONS = (
    "line", "parabola", "polynomial", "gaussian", "lorentzian",
    "exponential", "sigmoid", "error function",
)
TEXT_DATA_FILETYPES = [
    ("CSV data", "*.csv"),
    ("Text data", "*.txt"),
]
RASTER_DATA_FILETYPES = [
    ("PNG image", "*.png"),
    ("JPEG image", "*.jpg *.jpeg"),
]


@dataclass
class DataSet:
    kind: str
    data: np.ndarray
    axes: list[np.ndarray]
    title: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    units: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent: str | None = None
    derivation: dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    kind: str
    confidence: float
    shape: tuple[int, ...]
    details: str
    parser: str


def _lines(path: str) -> list[str]:
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return Path(path).read_text(encoding=enc).splitlines()
        except UnicodeDecodeError:
            pass
    raise ValueError("Could not decode text file")


def _nums(line: str) -> list[float]:
    try:
        return [float(x) for x in re.split(r"[\t,; ]+", line.strip()) if x]
    except ValueError:
        return []


def _meta(lines: list[str]) -> tuple[dict[str, str], int]:
    out, first = {}, len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_comment = stripped.startswith("#")
        s = stripped.lstrip("#").strip()
        if not s:
            continue
        vals = _nums(s)
        if vals and not is_comment:
            first = i
            break
        if "=" in s:
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip().lstrip("=\t ")
    return out, first


def detect_text(path: str) -> list[Detection]:
    """Return ranked, explainable interpretations without loading full arrays."""
    lines = _lines(path)
    meta, start = _meta(lines)
    rows = []
    for line in lines[start:start + 20]:
        values = _nums(line)
        if values:
            rows.append(values)
    detections: list[Detection] = []
    if not rows:
        return [Detection("curve", .05, (0,), "No numeric table found", "generic")]
    widths = [len(r) for r in rows]
    if meta.get("HyperScopeDataVersion") == "1":
        kind = meta.get("HyperScopeKind", "")
        try:
            shape = tuple(int(value) for value in meta["HyperScopeShape"].split(","))
        except (KeyError, ValueError) as error:
            raise ValueError("Invalid HyperScope shape metadata") from error
        if kind not in {"curve", "image", "hyperspectral"}:
            raise ValueError(f"Invalid HyperScope data kind: {kind!r}")
        detections.append(Detection(
            kind, 1.0, shape, "Self-describing HyperScope data export",
            "hyperscope_export",
        ))
    if all(k in meta for k in ("XPoints", "YPoints")):
        sh = (int(meta["YPoints"]), int(meta["XPoints"]))
        detections.append(Detection("image", .99, sh, "XPoints/YPoints metadata plus coordinate rows", "coordinate_image"))
    first = rows[0]
    if len(first) >= 3 and all(abs(v-round(v)) < 1e-9 for v in first[:3]):
        nx, ny, nz = map(int, first[:3])
        if nx > 1 and ny > 1 and nz > 2:
            detections.append(Detection(
                "hyperspectral", .99, (ny, nx, nz),
                "First row declares X, Y, Z sizes",
                "declared_hyperspectral",
            ))
    # Coordinate hyperspectral images use an axis row then X,Y,curve rows.
    if len(widths) > 1 and widths[0] > 3 and all(w == widths[0] + 2 for w in widths[1:]):
        ncol = widths[0] + 2
        # Coordinate + curve tables have x,y then z samples.
        ndata = len(lines) - start - 1
        nside = int(round(np.sqrt(max(ndata, 0))))
        if ndata > 0 and nside * nside == ndata:
            detections.append(Detection("hyperspectral", .93, (nside, nside, ncol-2),
                                        "Axis row followed by square XY coordinate/curve grid",
                                        "coordinate_hyperspectral"))
    if len(rows) == 1 and widths[0] > 3:
        detections.append(Detection(
            "curve", .95, (widths[0],),
            "One numeric row; import as a row-based Curve", "row_line",
        ))
    elif len(set(widths)) == 1 and widths[0] > 3:
        detections.append(Detection(
            "curve", .72, (widths[0],),
            "A selected row may be imported as a row-based Curve",
            "row_line",
        ))
    if len(set(widths)) == 1 and widths[0] == 2:
        detections.append(Detection("curve", .96, (len(lines)-start, 2), "Two numeric columns", "two_column"))
    if len(rows) > 1 and len(set(widths)) == 1 and widths[0] == 1:
        detections.append(Detection(
            "curve", .90, (len(lines)-start,),
            "One numeric column with an implicit index axis", "one_column",
        ))
    if len(set(widths)) == 1 and widths[0] in (3, 4):
        detections.append(Detection("image", .78, (len(lines)-start,), "Coordinate-value table", "coordinate_image"))
    if len(rows) > 1 and len(set(widths)) == 1 and widths[0] > 2:
        detections.append(Detection(
            "image", .82, (len(lines)-start, widths[0]),
            "Rectangular numeric matrix", "matrix",
        ))
    if not detections:
        detections.append(Detection("image", .45, (len(lines)-start, max(widths)), "Generic rectangular numeric table", "matrix"))
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


def _axis(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def _dataset_axis_sizes(kind: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    if kind == "curve" and len(shape) == 1:
        return shape
    if kind == "image" and len(shape) == 2:
        return (shape[1], shape[0])
    if kind == "hyperspectral" and len(shape) == 3:
        return (shape[1], shape[0], shape[2])
    raise ValueError(f"Invalid {kind!r} data shape: {shape}")


def load_text(path: str, det: Detection, row_index: int = 0) -> DataSet:
    lines = _lines(path)
    meta, start = _meta(lines)
    title = Path(path).stem
    if det.parser == "hyperscope_export":
        try:
            shape = tuple(int(value) for value in meta["HyperScopeShape"].split(","))
            labels = json.loads(meta.get("HyperScopeLabels", "[]"))
            units = json.loads(meta.get("HyperScopeUnits", "[]"))
            saved_title = json.loads(meta.get("HyperScopeTitle", json.dumps(title)))
            axes = [
                np.asarray(json.loads(meta[f"HyperScopeAxis{index}"]), dtype=float)
                for index in range(len(shape))
            ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Invalid HyperScope export metadata") from error
        numeric = np.asarray(
            [row for row in (_nums(line) for line in lines[start:]) if row],
            dtype=float,
        )
        payload = numeric[:, 1] if det.kind == "curve" else numeric.ravel()
        if payload.size != int(np.prod(shape)):
            raise ValueError(
                f"HyperScope payload has {payload.size} values; expected {int(np.prod(shape))}"
            )
        data = payload.reshape(shape)
        axis_sizes = _dataset_axis_sizes(det.kind, shape)
        if any(len(axis) != size for axis, size in zip(axes, axis_sizes)):
            raise ValueError("HyperScope axis lengths do not match the saved data shape")
        return DataSet(
            det.kind, data, axes, saved_title, path, meta, units, labels
        )
    if det.parser == "declared_hyperspectral":
        head = _nums(lines[start])
        nx, ny, nz = map(int, head[:3])
        z = np.asarray(_nums(lines[start+1])[:nz])
        table = np.loadtxt(lines[start+2:], ndmin=2)
        hyperspectral = table[:, -nz:].reshape(ny, nx, nz)
        return DataSet("hyperspectral", hyperspectral, [_axis(nx), _axis(ny), z], title, path, meta,
                       ["index", "index", ""], ["X", "Y", "Spectral"])
    if det.parser == "coordinate_hyperspectral":
        axisrow = np.asarray(_nums(lines[start]))
        table = np.loadtxt(lines[start+1:], ndmin=2)
        x, y = np.unique(table[:, 0]), np.unique(table[:, 1])
        z = axisrow[-(table.shape[1]-2):]
        hyperspectral = table[:, 2:].reshape(len(y), len(x), len(z))
        return DataSet("hyperspectral", hyperspectral, [x, y, z], title, path, meta,
                       ["µm", "µm", "cm⁻¹"], ["X", "Y", "Raman shift"])
    numeric = [r for r in (_nums(x) for x in lines[start:]) if r]
    if det.parser == "row_line":
        a = np.asarray(numeric[max(0, min(row_index, len(numeric)-1))])
        return DataSet("curve", a, [_axis(len(a))], title, path, meta, ["index", "a.u."], ["X", "Signal"],
                       derivation={"row": row_index})
    a = np.asarray(numeric, float)
    if det.kind == "curve":
        if det.parser == "one_column":
            values = a[:, 0]
            return DataSet(
                "curve", values, [_axis(len(values))], title, path, meta,
                ["index", "a.u."], ["X", "Signal"],
            )
        return DataSet("curve", a[:, 1], [a[:, 0]], title, path, meta,
                       ["", "a.u."], ["X", "Signal"])
    if det.parser == "coordinate_image":
        nx = int(meta.get("XPoints", len(np.unique(a[:, 0]))))
        ny = int(meta.get("YPoints", len(np.unique(a[:, 1]))))
        image = a[:, -1].reshape(ny, nx)
        xs = np.linspace(float(meta.get("XOrigin", 0)), float(meta.get("XOrigin", 0))+float(meta.get("XSize", nx-1)), nx)
        ys = np.linspace(float(meta.get("YOrigin", 0)), float(meta.get("YOrigin", 0))+float(meta.get("YSize", ny-1)), ny)
        return DataSet("image", image, [xs, ys], title, path, meta,
                       [meta.get("XUnits", ""), meta.get("YUnits", ""), meta.get("ZUnits", "")], ["X", "Y", "Value"])
    return DataSet("image", a, [_axis(a.shape[1]), _axis(a.shape[0])], title, path, meta,
                   ["index", "index", "a.u."], ["X", "Y", "Value"])


def export_dataset_text(dataset: DataSet, path: str) -> None:
    """Write a lossless, self-describing CSV or tab-delimited data file."""
    suffix = Path(path).suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise ValueError("Text data exports must use a .csv or .txt extension")
    shape = tuple(int(size) for size in dataset.data.shape)
    axis_sizes = _dataset_axis_sizes(dataset.kind, shape)
    if len(dataset.axes) != len(axis_sizes):
        raise ValueError("The number of axes does not match the data dimensions")
    if any(len(axis) != size for axis, size in zip(dataset.axes, axis_sizes)):
        raise ValueError("Axis lengths do not match the data shape")
    header = [
        "HyperScopeDataVersion = 1",
        f"HyperScopeKind = {dataset.kind}",
        "HyperScopeShape = " + ",".join(str(size) for size in shape),
        "HyperScopeTitle = " + json.dumps(dataset.title, ensure_ascii=False),
        "HyperScopeLabels = " + json.dumps(dataset.labels, ensure_ascii=False),
        "HyperScopeUnits = " + json.dumps(dataset.units, ensure_ascii=False),
    ]
    header.extend(
        f"HyperScopeAxis{index} = "
        + json.dumps(np.asarray(axis, dtype=float).tolist())
        for index, axis in enumerate(dataset.axes)
    )
    if dataset.kind == "curve":
        payload = np.column_stack((dataset.axes[0], dataset.data))
    elif dataset.kind in {"image", "hyperspectral"}:
        payload = dataset.data.reshape(dataset.data.shape[0], -1)
    else:
        raise ValueError(f"Unsupported data kind: {dataset.kind!r}")
    np.savetxt(
        path,
        payload,
        delimiter="," if suffix == ".csv" else "\t",
        header="\n".join(header),
        comments="# ",
    )


def evaluate_fit_model(
    x: np.ndarray, model: str, parameters: np.ndarray
) -> np.ndarray:
    """Evaluate fitted model parameters at arbitrary X coordinates."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if model == "polynomial":
        return np.polyval(p, x)
    functions = {
        "line": lambda x, a, b: a*x+b,
        "parabola": lambda x, a, b, c: a*x*x+b*x+c,
        "gaussian": lambda x, a, c, s, b: b+a*np.exp(-.5*((x-c)/s)**2),
        "lorentzian": lambda x, a, c, g, b: b+a*g*g/((x-c)**2+g*g),
        "exponential": lambda x, a, t, b: b+a*np.exp(np.clip(x/t, -100, 100)),
        "sigmoid": lambda x, a, c, s, b: b+a/(1+np.exp(np.clip(-(x-c)/s, -100, 100))),
        "error function": lambda x, a, c, s, b: b+a*special.erf((x-c)/s),
    }
    try:
        function = functions[model]
    except KeyError as error:
        raise ValueError(f"Unknown fit function: {model}") from error
    return np.asarray(function(x, *p), dtype=float)


def fit_curve(x: np.ndarray, y: np.ndarray, model: str, order: int = 3):
    if model == "polynomial":
        p = np.polyfit(x, y, order)
        return evaluate_fit_model(x, model, p), p
    initial_parameters = {
        "line": [1, np.mean(y)],
        "parabola": [0, 1, np.mean(y)],
        "gaussian": [np.ptp(y), x[np.argmax(y)], np.ptp(x)/8, np.min(y)],
        "lorentzian": [np.ptp(y), x[np.argmax(y)], np.ptp(x)/8, np.min(y)],
        "exponential": [np.ptp(y), max(np.ptp(x), 1), np.min(y)],
        "sigmoid": [np.ptp(y), np.mean(x), np.ptp(x)/8, np.min(y)],
        "error function": [np.ptp(y)/2, np.mean(x), np.ptp(x)/8, np.mean(y)],
    }
    try:
        p0 = initial_parameters[model]
    except KeyError as error:
        raise ValueError(f"Unknown fit function: {model}") from error
    f = lambda values, *parameters: evaluate_fit_model(values, model, parameters)
    p, _ = optimize.curve_fit(f, x, y, p0=p0, maxfev=30000)
    return evaluate_fit_model(x, model, p), p


def profile_cursor_visibility(show_auxiliary: bool) -> dict[str, bool]:
    """The curve cursor is always active; the checkbox controls auxiliary slices."""
    return {
        "curve": True,
        "xz": bool(show_auxiliary),
        "yz": bool(show_auxiliary),
    }


def initial_hyperspectral_position(
    shape: tuple[int, int, int], z_axis: np.ndarray
) -> list[int]:
    """Center spatially and select the Z value drawn at the graph's right edge."""
    ny, nx, nz = shape
    if min(shape) < 1:
        raise ValueError("Hyperspectral data dimensions must be non-empty")
    z_axis = np.asarray(z_axis, dtype=float)
    if z_axis.shape != (nz,):
        raise ValueError("The Z axis length does not match the hyperspectral data")
    if not np.all(np.isfinite(z_axis)):
        raise ValueError("The Z axis must contain finite values")
    return [nx // 2, ny // 2, int(np.argmax(z_axis))]


def metadata_sidebar_text(
    dataset: DataSet,
    view_information: dict[str, Any] | None = None,
    include_metadata: bool = True,
) -> str:
    """Format view information first and file metadata last."""
    view_names = {
        "curve": "Curve",
        "image": "2D Image",
        "hyperspectral": "Hyperspectral Image",
    }
    lines = [
        f"View type: {view_names.get(dataset.kind, dataset.kind.title())}",
        f"Shape: {dataset.data.shape}",
        f"Units: {dataset.units}",
    ]
    if view_information:
        for key, value in view_information.items():
            lines.append(
                f"\n{key}" if value is None else f"{key}: {value}"
            )
    if include_metadata:
        lines.extend(["", "Metadata:"])
        if dataset.metadata:
            lines.extend(f"{key}: {value}" for key, value in dataset.metadata.items())
        else:
            lines.append("None")
    return "\n".join(lines)


def hyperspectral_sections(data: np.ndarray, ix: int, iy: int, iz: int) -> dict[str, np.ndarray]:
    """Extract every hyperspectral view from one unambiguous cursor coordinate.

    Storage order is (y, x, z); returned image slices use vertical, horizontal
    order expected by imshow: XY=(y,x), XZ=(z,x), and YZ=(z,y).
    """
    if data.ndim != 3:
        raise ValueError("Hyperspectral data must have shape (y, x, z)")
    ny, nx, nz = data.shape
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        raise IndexError("Cursor coordinate is outside the hyperspectral image")
    return {
        "xy": data[:, :, iz],
        "xz": data[iy, :, :].T,
        "yz": data[:, ix, :],
        "x_profile": data[iy, :, iz],
        "y_profile": data[:, ix, iz],
        "z_profile": data[iy, ix, :],
    }


def orient_spectral_planes(
    sections: dict[str, np.ndarray],
    z_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Put increasing Z toward the top of XZ and right side of YZ."""
    z_axis = np.asarray(z_axis, dtype=float)
    xz = np.asarray(sections["xz"])
    yz = np.asarray(sections["yz"])
    if z_axis.ndim != 1 or xz.shape[0] != len(z_axis) or yz.shape[1] != len(z_axis):
        raise ValueError("Planar slice dimensions must match the Z axis")
    if not len(z_axis) or not np.all(np.isfinite(z_axis)):
        raise ValueError("The Z axis must be non-empty and finite")
    order = np.argsort(z_axis, kind="stable")
    limits = (float(z_axis[order[0]]), float(z_axis[order[-1]]))
    return xz[order, :], yz[:, order], limits


def contrast_reference_limits(data: np.ndarray, iz: int, reference) -> tuple[float, float] | None:
    """Return finite contrast limits from an inclusive XY reference rectangle."""
    return image_contrast_limits(data[:, :, iz], reference)


def image_contrast_limits(values: np.ndarray, reference) -> tuple[float, float] | None:
    """Return finite display limits from an inclusive 2D reference rectangle."""
    if reference is None:
        return None
    (x0, y0), (x1, y1) = reference
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    xa, xb = max(0, xa), min(values.shape[1] - 1, xb)
    ya, yb = max(0, ya), min(values.shape[0] - 1, yb)
    region = values[ya:yb + 1, xa:xb + 1]
    finite = region[np.isfinite(region)]
    if not finite.size:
        return None
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo), 1.0) * 1e-12
        lo, hi = lo - pad, hi + pad
    return lo, hi


def normalize_display_channel(
    values: np.ndarray,
    reference=None,
    use_reference: bool = True,
) -> np.ndarray:
    """Normalize a scalar image, optionally using an inclusive XY reference.

    Values outside the reference range are clipped. NaNs remain finite in the
    returned display image so Matplotlib receives a valid RGB array.
    """
    values = np.asarray(values, dtype=float)
    basis = values
    if use_reference and reference is not None:
        (x0, y0), (x1, y1) = reference
        xa, xb = sorted((int(x0), int(x1)))
        ya, yb = sorted((int(y0), int(y1)))
        xa, xb = max(0, xa), min(values.shape[1] - 1, xb)
        ya, yb = max(0, ya), min(values.shape[0] - 1, yb)
        basis = values[ya:yb + 1, xa:xb + 1]
    finite = basis[np.isfinite(basis)]
    if not finite.size:
        return np.zeros_like(values)
    low, high = float(np.min(finite)), float(np.max(finite))
    if low == high:
        return np.zeros_like(values)
    normalized = (values - low) / (high - low)
    return np.nan_to_num(np.clip(normalized, 0, 1))


def orient_image_for_export(image: np.ndarray) -> np.ndarray:
    """Match a raster file's top row to an on-screen image using origin='lower'."""
    values = np.asarray(image)
    if values.ndim not in {2, 3}:
        raise ValueError("Only scalar or RGB images can be exported")
    return np.flip(values, axis=0).copy()


def save_display_image(
    path: str,
    image: np.ndarray,
    colormap: str | None = None,
) -> None:
    """Save a scalar or RGB display image with its on-screen orientation."""
    suffix = Path(path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Display images must use PNG or JPEG")
    import matplotlib.pyplot as plt

    values = np.asarray(image)
    options = {"cmap": colormap} if values.ndim == 2 and colormap else {}
    plt.imsave(path, orient_image_for_export(values), **options)


def display_colormap(name: str, inverted: bool) -> str:
    """Return Matplotlib's normal or reversed colormap name."""
    if not inverted:
        return name
    base_name = name[:-2] if name.endswith("_r") else f"{name}_r"
    if base_name not in colormaps:
        raise ValueError(f"Colormap {name!r} does not have a registered reverse")
    return base_name


def grouped_colormaps() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return non-reversed Matplotlib maps in documentation category order."""
    documented = (
        ("Perceptually Uniform Sequential",
         ("viridis", "plasma", "inferno", "magma", "cividis")),
        ("Sequential",
         ("Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
          "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu",
          "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn")),
        ("Sequential (2)",
         ("gray", "bone", "pink", "spring", "summer", "autumn", "winter",
          "cool", "Wistia", "hot", "afmhot", "gist_heat", "copper")),
        ("Diverging",
         ("PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu",
          "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic", "berlin",
          "managua", "vanimo")),
        ("Cyclic", ("twilight", "twilight_shifted", "hsv")),
        ("Qualitative",
         ("Pastel1", "Pastel2", "Paired", "Accent", "okabe_ito", "Dark2",
          "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c")),
        ("Miscellaneous",
         ("flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
          "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg",
          "gist_rainbow", "rainbow", "jet", "turbo", "nipy_spectral",
          "gist_ncar")),
    )
    available = {name for name in colormaps if not name.endswith("_r")}
    listed = {name for _, names in documented for name in names}
    groups = [
        (category, tuple(name for name in names if name in available))
        for category, names in documented
    ]
    extras = tuple(sorted(available - listed))
    if extras:
        category, names = groups[-1]
        groups[-1] = (category, names + extras)
    return tuple(groups)


def default_curve_style() -> dict[str, Any]:
    """Return keyword names accepted directly by Matplotlib Line2D."""
    return {
        "color": "#1565c0", "linewidth": 1.5, "linestyle": "-",
        "marker": "", "markersize": 6.0, "markerfacecolor": "#1565c0",
        "markeredgecolor": "#1565c0",
    }


def default_fit_style() -> dict[str, Any]:
    """Return the default fit appearance, drawn prominently in orange."""
    style = default_curve_style()
    style.update({
        "color": "orange", "linewidth": 2.0,
        "markerfacecolor": "orange", "markeredgecolor": "orange",
    })
    return style


def curve_style_from_controls(
    line_style: str,
    line_width: str,
    line_color: str,
    marker: str,
    marker_size: str,
    marker_color: str,
) -> dict[str, Any]:
    """Translate user-facing dropdown values into Matplotlib line keywords."""
    if line_style not in CURVE_LINE_STYLES:
        raise ValueError("Unknown line style")
    if marker not in CURVE_MARKERS:
        raise ValueError("Unknown marker")
    width, size = float(line_width), float(marker_size)
    if width <= 0 or size <= 0:
        raise ValueError("Line width and marker size must be positive")
    return {
        "linestyle": "" if line_style == "None" else line_style,
        "linewidth": width,
        "color": line_color,
        "marker": "" if marker == "None" else marker,
        "markersize": size,
        "markerfacecolor": marker_color,
        "markeredgecolor": marker_color,
    }


def curve_axis_scale(values: np.ndarray, logarithmic: bool) -> str:
    """Validate and return a Matplotlib scale for one Curve axis."""
    if not logarithmic:
        return "linear"
    values = np.asarray(values, dtype=float)
    if not values.size or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Logarithmic axes require finite values greater than zero")
    return "log"


def fit_curve_selection(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    parameter_text: str,
    between_markers: bool,
    cursors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate on selected data, then evaluate across the complete X axis."""
    if model not in CURVE_FIT_FUNCTIONS:
        raise ValueError(f"Unknown fit function: {model}")
    order = 3
    text = parameter_text.strip()
    if model == "polynomial":
        if text:
            value = text.split("=", 1)[-1].strip()
            order = int(value)
        if not 1 <= order <= 20:
            raise ValueError("Polynomial order must be between 1 and 20")
    elif text:
        raise ValueError(f"{model.title()} does not require parameters")
    low, high = (
        sorted((int(cursors[0]), int(cursors[1])))
        if between_markers else (0, len(y) - 1)
    )
    _, parameters = fit_curve(
        np.asarray(x)[low:high + 1],
        np.asarray(y)[low:high + 1],
        model,
        order,
    )
    full_fit = evaluate_fit_model(np.asarray(x), model, parameters)
    return full_fit, np.asarray(parameters)


def labeled_fit_parameters(
    model: str, parameters: np.ndarray
) -> dict[str, float]:
    """Attach meaningful model-specific names to fitted parameters."""
    parameter_names = {
        "line": ("Slope", "Intercept"),
        "parabola": (
            "Quadratic Coefficient", "Linear Coefficient", "Intercept"
        ),
        "gaussian": ("Amplitude", "Center", "Sigma", "Baseline"),
        "lorentzian": ("Amplitude", "Center", "Gamma", "Baseline"),
        "exponential": ("Amplitude", "Scale", "Baseline"),
        "sigmoid": ("Amplitude", "Center", "Scale", "Baseline"),
        "error function": ("Amplitude", "Center", "Scale", "Baseline"),
    }
    values = np.ravel(parameters)
    if model == "polynomial":
        degree = len(values) - 1
        names = tuple(
            "Intercept" if power == 0 else
            ("Linear Coefficient" if power == 1 else f"X^{power} Coefficient")
            for power in range(degree, -1, -1)
        )
    else:
        try:
            names = parameter_names[model]
        except KeyError as error:
            raise ValueError(f"Unknown fit function: {model}") from error
    if len(names) != len(values):
        raise ValueError("Fit parameter count does not match the selected model")
    return {name: float(value) for name, value in zip(names, values)}


def fit_sidebar_information(
    show_fit: bool,
    model: str,
    fit_line: np.ndarray | None,
    parameters: dict[str, float] | None,
) -> dict[str, Any]:
    """Return a spaced fit section only when fit visibility is enabled."""
    if not show_fit:
        return {}
    information: dict[str, Any] = {
        "Fit": None,
        "Fit Function": model if fit_line is not None else "None",
        "Fit Parameters": None,
    }
    if parameters is None:
        information["Fit Status"] = "No fit"
    else:
        information.update(parameters)
    return information


def curve_cursor_statistics(
    x: np.ndarray,
    y: np.ndarray,
    cursors: list[int] | tuple[int, int],
    cursors_visible: bool = True,
) -> dict[str, float]:
    """Calculate cursor-range or whole-curve statistics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    first, second = (int(cursors[0]), int(cursors[1]))
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("Curve X and Y data must be non-empty, equal-length vectors")
    if not (0 <= first < len(x) and 0 <= second < len(x)):
        raise ValueError("Curve cursor index is outside the data")
    if cursors_visible:
        low, high = sorted((first, second))
        values = y[low:high + 1]
        x_difference = float(x[second] - x[first])
        y_difference = float(y[second] - y[first])
        differences = {
            "X Difference": x_difference,
            "Y Difference": y_difference,
            "Distance": float(np.hypot(x_difference, y_difference)),
        }
    else:
        values = y
        differences = {
            "X Range": float(np.nanmax(x) - np.nanmin(x)),
            "Y Range": float(np.nanmax(y) - np.nanmin(y)),
        }
    return {
        **differences,
        "Average": float(np.nanmean(values)),
        "Minimum": float(np.nanmin(values)),
        "Maximum": float(np.nanmax(values)),
    }


def curve_sidebar_information(
    x: np.ndarray,
    y: np.ndarray,
    cursors: list[int],
    cursor_visibility: list[bool],
) -> dict[str, Any]:
    """Build plain-Python sidebar values for the visible curve cursors."""
    information: dict[str, Any] = {}
    for index, visible in enumerate(cursor_visibility):
        if visible:
            cursor = cursors[index]
            information[f"Cursor {index + 1} ({CURVE_CURSOR_COLORS[index]})"] = (
                float(x[cursor]), float(y[cursor])
            )
    both_visible = all(cursor_visibility)
    information["Between Cursors" if both_visible else "Across Data"] = ""
    information.update(
        curve_cursor_statistics(x, y, cursors, both_visible)
    )
    return information


def curve_range_dataset(source: DataSet, cursors: list[int]) -> DataSet:
    """Create an inclusive child curve between the two cursor indices."""
    if source.kind != "curve":
        raise ValueError("Cursor ranges can only be extracted from curves")
    low, high = sorted((int(cursors[0]), int(cursors[1])))
    if not (0 <= low <= high < len(source.data)):
        raise ValueError("Curve cursor index is outside the data")
    return DataSet(
        "curve",
        source.data[low:high + 1].copy(),
        [source.axes[0][low:high + 1].copy()],
        f"{source.title} cursor range",
        source.source,
        {},
        list(source.units),
        list(source.labels),
        parent=source.uid,
        derivation={
            "start_index": low,
            "end_index": high,
            "start_x": float(source.axes[0][low]),
            "end_x": float(source.axes[0][high]),
        },
    )


def duplicate_dataset(source: DataSet) -> DataSet:
    """Create an independent child dataset with copied arrays and metadata."""
    return DataSet(
        source.kind,
        source.data.copy(),
        [axis.copy() for axis in source.axes],
        f"{source.title} copy",
        source.source,
        copy.deepcopy(source.metadata),
        list(source.units),
        list(source.labels),
        parent=source.uid,
        derivation={"operation": "duplicate"},
    )


def dataset_with_replaced_axis(axis_source: DataSet, target: DataSet) -> DataSet:
    """Create a child Curve or hyperspectral view using a Curve's X axis."""
    if axis_source.kind != "curve":
        raise ValueError("The axis source must be a Curve")
    if target.kind not in {"curve", "hyperspectral"}:
        raise ValueError("X axes can only be sent to Curve or hyperspectral views")
    target_axis_index = 0 if target.kind == "curve" else 2
    required_size = (
        len(target.data)
        if target.kind == "curve" else target.data.shape[2]
    )
    if len(axis_source.axes[0]) != required_size:
        raise ValueError(
            f"Axis has {len(axis_source.axes[0])} values; "
            f"the target requires {required_size}"
        )
    axes = [axis.copy() for axis in target.axes]
    axes[target_axis_index] = axis_source.axes[0].copy()
    labels = list(target.labels)
    units = list(target.units)
    labels[target_axis_index] = axis_source.labels[0]
    units[target_axis_index] = axis_source.units[0]
    return DataSet(
        target.kind,
        target.data.copy(),
        axes,
        f"{target.title} with {axis_source.title} axis",
        target.source,
        copy.deepcopy(target.metadata),
        units,
        labels,
        parent=target.uid,
        derivation={
            "operation": "replace_axis",
            "axis_source": axis_source.uid,
            "target_axis": "X" if target.kind == "curve" else "Z",
        },
    )


def raster_export_array(dataset: DataSet) -> np.ndarray:
    """Return 2D image data for a lossless raster-oriented export."""
    if dataset.kind != "image":
        raise ValueError("Raster data export is only available for 2D images")
    data = np.asarray(dataset.data)
    if data.ndim != 2:
        raise ValueError("Image data must be two-dimensional")
    return data


def data_export_filetypes(kind: str) -> list[tuple[str, str]]:
    """Return only data formats that faithfully represent the dataset kind."""
    if kind == "image":
        return [*TEXT_DATA_FILETYPES, *RASTER_DATA_FILETYPES]
    if kind in {"curve", "hyperspectral"}:
        return list(TEXT_DATA_FILETYPES)
    raise ValueError(f"Unsupported data kind: {kind!r}")


def collapse_spectral_range(chunk: np.ndarray, method: str) -> np.ndarray:
    """Collapse the last axis of a spectral range using a named statistic."""
    methods = {
        "Average": np.nanmean,
        "Maximum": np.nanmax,
        "Minimum": np.nanmin,
    }
    try:
        reducer = methods[method]
    except KeyError as error:
        raise ValueError(f"Unknown range value method: {method}") from error
    return reducer(chunk, axis=-1)


def hyperspectral_slice_dataset(dataset: DataSet, iz: int) -> DataSet:
    """Create a standalone 2D dataset from one hyperspectral Z coordinate."""
    if dataset.kind != "hyperspectral" or dataset.data.ndim != 3:
        raise ValueError("A 2D slice requires a hyperspectral image")
    if not 0 <= iz < dataset.data.shape[2]:
        raise IndexError("Z slice index is outside the hyperspectral image")
    z_value = float(dataset.axes[2][iz])
    return DataSet(
        "image",
        dataset.data[:, :, iz].copy(),
        [dataset.axes[0].copy(), dataset.axes[1].copy()],
        f"{dataset.title} 2D slice",
        dataset.source,
        {},
        [dataset.units[0], dataset.units[1], "a.u."],
        [dataset.labels[0], dataset.labels[1], "Signal"],
        parent=dataset.uid,
        derivation={"z_index": iz, "z_value": z_value},
    )


def image_profile_dataset(
    dataset: DataSet,
    axis: str,
    ix: int,
    iy: int,
) -> DataSet:
    """Create the horizontal or vertical cursor profile of a 2D image."""
    if dataset.kind != "image" or dataset.data.ndim != 2:
        raise ValueError("An image profile requires a 2D image")
    ny, nx = dataset.data.shape
    if not 0 <= ix < nx or not 0 <= iy < ny:
        raise IndexError("Profile cursor is outside the image")
    if axis == "x":
        values = dataset.data[iy, :]
        coordinates = dataset.axes[0]
        axis_index = 0
        derivation = {"axis": "x", "y_index": iy}
    elif axis == "y":
        values = dataset.data[:, ix]
        coordinates = dataset.axes[1]
        axis_index = 1
        derivation = {"axis": "y", "x_index": ix}
    else:
        raise ValueError("Image profile axis must be 'x' or 'y'")
    value_unit = dataset.units[2] if len(dataset.units) > 2 else "a.u."
    value_label = dataset.labels[2] if len(dataset.labels) > 2 else "Signal"
    return DataSet(
        "curve",
        np.asarray(values).copy(),
        [np.asarray(coordinates).copy()],
        f"{dataset.title} {axis.upper()} profile",
        dataset.source,
        {},
        [dataset.units[axis_index], value_unit],
        [dataset.labels[axis_index], value_label],
        parent=dataset.uid,
        derivation=derivation,
    )


def image_line_profile(
    dataset: DataSet,
    first: list[int] | tuple[int, int],
    second: list[int] | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a 2D image along the segment between two pixel cursors."""
    if dataset.kind != "image" or dataset.data.ndim != 2:
        raise ValueError("A line profile requires a 2D image")
    ix, iy = map(int, first)
    jx, jy = map(int, second)
    ny, nx = dataset.data.shape
    if not (
        0 <= ix < nx and 0 <= jx < nx
        and 0 <= iy < ny and 0 <= jy < ny
    ):
        raise IndexError("Line-profile cursor is outside the image")
    count = max(abs(jx - ix), abs(jy - iy), 2) + 1
    values = ndimage.map_coordinates(
        dataset.data,
        [np.linspace(iy, jy, count), np.linspace(ix, jx, count)],
        order=1,
    )
    x_axis, y_axis = dataset.axes
    distance = np.linspace(
        0.0,
        np.hypot(x_axis[jx] - x_axis[ix], y_axis[jy] - y_axis[iy]),
        count,
    )
    return distance, values


def consume_first_use(seen: set[str], tool_name: str) -> bool:
    """Return True once for a tool, then suppress its instructional prompt."""
    if tool_name in seen:
        return False
    seen.add(tool_name)
    return True


def nearest_axis_index(axis: np.ndarray, coordinate: float) -> int:
    """Return the closest coordinate index for ascending or descending axes."""
    values = np.asarray(axis, dtype=float)
    if not values.size or not np.isfinite(coordinate):
        raise ValueError("A finite coordinate and non-empty axis are required")
    return int(np.nanargmin(np.abs(values - coordinate)))


def validate_session_payload(payload: Any) -> None:
    """Fail clearly before changing the active session."""
    if not isinstance(payload, dict):
        raise ValueError("Session payload is not a dictionary")
    if payload.get("version") != APP_VERSION:
        raise ValueError(
            f"Unsupported session version {payload.get('version')!r}; "
            f"this prototype opens version {APP_VERSION} only"
        )
    if not isinstance(payload.get("datasets"), dict):
        raise ValueError("Session datasets are missing or invalid")
    if not isinstance(payload.get("states"), dict):
        raise ValueError("Session view states are missing or invalid")


def dataset_from_array(data: np.ndarray, *, kind: str | None=None, axes: list[np.ndarray] | None=None, title: str='Notebook data', metadata: dict[str, Any] | None=None, units: list[str] | None=None, labels: list[str] | None=None) -> DataSet:
    """Build a validated HyperScope dataset from a 1D, 2D, or 3D ndarray."""
    values = np.asarray(data)
    inferred = {1: 'curve', 2: 'image', 3: 'hyperspectral'}.get(values.ndim)
    kind = kind or inferred
    if kind != inferred:
        raise ValueError(f'{kind!r} does not match a {values.ndim}D array')
    if kind == 'curve':
        sizes = (values.shape[0],)
    elif kind == 'image':
        sizes = (values.shape[1], values.shape[0])
    else:
        sizes = (values.shape[1], values.shape[0], values.shape[2])
    axes = [np.arange(size, dtype=float) for size in sizes] if axes is None else axes
    if len(axes) != len(sizes) or any((len(axis) != size for axis, size in zip(axes, sizes))):
        raise ValueError('Axis lengths do not match the data shape')
    default_labels = {'curve': ['X', 'Signal'], 'image': ['X', 'Y', 'Value'], 'hyperspectral': ['X', 'Y', 'Z']}[kind]
    default_units = {'curve': ['index', 'a.u.'], 'image': ['index', 'index', 'a.u.'], 'hyperspectral': ['index', 'index', 'index']}[kind]
    return DataSet(kind, values.copy(), [np.asarray(a, float).copy() for a in axes], title, 'notebook', copy.deepcopy(metadata or {}), list(units or default_units), list(labels or default_labels))

def import_data(path: str, *, kind: str | None=None, row_index: int=0) -> DataSet:
    """Detect and import a HyperScope text file without opening a Tk window."""
    detections = detect_text(path)
    candidates = [item for item in detections if kind is None or item.kind == kind]
    if not candidates:
        raise ValueError(f'No {kind!r} interpretation is available')
    return load_text(path, candidates[0], row_index)

def import_numpy(path: str, **kwargs) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Import text directly to data, axes, and metadata copies."""
    dataset = import_data(path, **kwargs)
    return (dataset.data.copy(), [axis.copy() for axis in dataset.axes], copy.deepcopy(dataset.metadata))

def _widgets():
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise ImportError(
            'Interactive controls require: '
            'python -m pip install ipywidgets ipympl'
        ) from error
    return (widgets, display)


def coordinate_slider_options(axis, unit=''):
    """Map displayed physical coordinates to stable integer array indices."""
    values = np.asarray(axis, dtype=float)
    suffix = f' {unit}' if unit else ''
    return [(f'{value:g}{suffix}', index) for index, value in enumerate(values)]


def _show_widget_canvas(view):
    """Display one live ipympl canvas with controls directly beneath it."""
    widgets, display = _widgets()
    canvas = view.figure.canvas
    if isinstance(canvas, widgets.Widget):
        view._display_widget = widgets.VBox([canvas, view.controls])
        display(view._display_widget)
    else:
        display(view.figure)
        display(view.controls)
    return view

class NotebookView:
    """Common extraction and persistence API for notebook views."""

    def __init__(self, dataset: DataSet):
        self.dataset = dataset
        self.figure = None
        self.controls = None

    def to_numpy(self, copy_array: bool=True) -> np.ndarray:
        return self.dataset.data.copy() if copy_array else self.dataset.data

    def to_scipy(self, copy_array: bool=True) -> np.ndarray:
        """Return the ndarray accepted directly by scipy.ndimage and scipy.signal."""
        return self.to_numpy(copy_array)

    def metadata(self) -> dict[str, Any]:
        return {'kind': self.dataset.kind, 'title': self.dataset.title, 'source': self.dataset.source, 'shape': self.dataset.data.shape, 'axes': [axis.copy() for axis in self.dataset.axes], 'labels': list(self.dataset.labels), 'units': list(self.dataset.units), 'metadata': copy.deepcopy(self.dataset.metadata), 'derivation': copy.deepcopy(self.dataset.derivation)}

    def profiles(self) -> dict[str, np.ndarray]:
        return {}

    def images(self) -> dict[str, np.ndarray]:
        return {}

    def extract(self) -> dict[str, Any]:
        return {'data': self.to_numpy(), 'metadata': self.metadata(), 'profiles': self.profiles(), 'images': self.images()}

    def save_numpy(self, path: str) -> None:
        path = str(path)
        if Path(path).suffix.lower() == '.npy':
            np.save(path, self.dataset.data)
            return
        if Path(path).suffix.lower() != '.npz':
            raise ValueError('NumPy output must use .npy or .npz')
        payload = {'data': self.dataset.data, 'metadata_json': np.asarray(json.dumps(self.metadata(), default=str))}
        payload.update({f'axis_{i}': axis for i, axis in enumerate(self.dataset.axes)})
        np.savez_compressed(path, **payload)

    def save_scipy(self, path: str) -> None:
        """Save data, axes, profiles, and images as a SciPy-written MAT file."""
        if Path(path).suffix.lower() != '.mat':
            raise ValueError('SciPy output must use .mat')
        from scipy.io import savemat
        payload = {'data': self.dataset.data, 'metadata_json': json.dumps(self.metadata(), default=str)}
        payload.update({f'axis_{i}': axis for i, axis in enumerate(self.dataset.axes)})
        payload.update({f'profile_{key}': value for key, value in self.profiles().items() if value is not None})
        payload.update({f'image_{key}': value for key, value in self.images().items()})
        savemat(path, payload, do_compression=True)

    def save_image(self, path: str, name: str='display') -> None:
        images = self.images()
        if name not in images:
            raise KeyError(f'Unknown image {name!r}; choose from {tuple(images)}')
        save_display_image(path, images[name])

    def show(self):
        _, display = _widgets()
        display(self.figure)
        display(self.controls)
        return self

class CurveNotebookView(NotebookView):

    def __init__(self, dataset: DataSet):
        super().__init__(dataset)
        if dataset.kind != 'curve':
            raise ValueError('Curve view requires 1D data')
        w, _ = _widgets()
        self._axis_values = [dataset.axes[0].copy(), dataset.data.copy()]
        self._axis_units = list(dataset.units[:2])
        n = len(dataset.data)
        self.cursors = [n // 3, 2 * n // 3]
        self.fit = None
        self.fit_parameters = {}
        self.data_color = '#1565c0'
        self.fit_color = 'orange'
        with plt.ioff():
            self.figure, self.axis = plt.subplots(figsize=(9, 5))
        self.figure.subplots_adjust(bottom=0.12)
        self.show_c1 = w.Checkbox(True, description='Cursor 1')
        self.show_c2 = w.Checkbox(True, description='Cursor 2')
        self.cursor_slider = w.SelectionRangeSlider(
            options=coordinate_slider_options(dataset.axes[0], dataset.units[0]),
            value=tuple(self.cursors), description='Cursors'
        )
        self.coordinates = w.HTML()
        self.fit_model = w.Dropdown(options=['line', 'parabola', 'polynomial', 'gaussian', 'lorentzian', 'exponential', 'sigmoid', 'error function'], description='Fit')
        self.fit_text = w.Text(description='Parameters')
        self.between = w.Checkbox(False, description='Between Markers')
        self.fit_button = w.Button(description='Fit')
        self.show_fit = w.Checkbox(True, description='Show Fit')
        self.fit_output = w.Output(
            layout={'max_height': '240px', 'overflow': 'auto'}
        )
        colors = ['#1565c0', 'orange', 'red', 'green', 'black', 'purple', 'cyan']
        self.data_color_widget = w.Dropdown(options=colors, value=self.data_color, description='Data color')
        self.fit_color_widget = w.Dropdown(options=colors, value=self.fit_color, description='Fit color')
        self.x_role = w.Dropdown(options=['Original X', 'Original Y'], description='X axis')
        self.y_role = w.Dropdown(options=['Original X', 'Original Y'], value='Original Y', description='Y axis')
        self.x_label = w.Text(value=dataset.labels[0], description='X label')
        self.y_label = w.Text(value=dataset.labels[1], description='Y label')
        self.log_x = w.Checkbox(False, description='Log X')
        self.log_y = w.Checkbox(False, description='Log Y')
        self.apply_axes = w.Button(description='Apply Axes')
        self.controls = w.VBox([w.HBox([self.show_c1, self.show_c2, self.cursor_slider, self.coordinates]), w.HBox([self.fit_model, self.fit_text, self.between, self.fit_button, self.show_fit]), self.fit_output, w.HBox([self.data_color_widget, self.fit_color_widget]), w.HBox([self.x_role, self.y_role, self.x_label, self.y_label, self.log_x, self.log_y, self.apply_axes])])
        for widget in (self.show_c1, self.show_c2, self.cursor_slider, self.show_fit, self.data_color_widget, self.fit_color_widget, self.x_label, self.y_label, self.log_x, self.log_y):
            widget.observe(self._changed, names='value')
        self.fit_button.on_click(self._fit)
        self.apply_axes.on_click(self._apply_axes)
        self.drag_cursor = None
        self._updating_cursor_slider = False
        self.figure.canvas.mpl_connect('button_press_event', self._press)
        self.figure.canvas.mpl_connect('motion_notify_event', self._motion)
        self.figure.canvas.mpl_connect('button_release_event', self._release)
        self._draw()

    def _changed(self, change=None):
        if self._updating_cursor_slider:
            return
        self.cursors = list(self.cursor_slider.value)
        self.data_color = self.data_color_widget.value
        self.fit_color = self.fit_color_widget.value
        self._draw()

    def _set_cursor(self, cursor, index):
        index = int(np.clip(index, 0, len(self.dataset.data) - 1))
        index = min(index, self.cursors[1]) if cursor == 0 else max(index, self.cursors[0])
        if index == self.cursors[cursor]:
            return
        self.cursors[cursor] = index
        self._updating_cursor_slider = True
        try:
            self.cursor_slider.value = tuple(self.cursors)
        finally:
            self._updating_cursor_slider = False
        self._draw()

    def _press(self, event):
        if event.inaxes != self.axis or event.xdata is None:
            return
        visible = [index for index, shown in enumerate((self.show_c1.value, self.show_c2.value)) if shown]
        if not visible:
            return
        data_index = nearest_axis_index(self.dataset.axes[0], event.xdata)
        self.drag_cursor = min(visible, key=lambda cursor: abs(data_index - self.cursors[cursor]))
        self._set_cursor(self.drag_cursor, data_index)

    def _motion(self, event):
        if self.drag_cursor is not None and event.inaxes == self.axis and event.xdata is not None:
            self._set_cursor(self.drag_cursor, nearest_axis_index(self.dataset.axes[0], event.xdata))

    def _release(self, event):
        self.drag_cursor = None

    def _fit(self, button=None):
        self.fit, pars = fit_curve_selection(self.dataset.axes[0], self.dataset.data, self.fit_model.value, self.fit_text.value, self.between.value, self.cursors)
        self.fit_parameters = labeled_fit_parameters(self.fit_model.value, pars)
        self.fit_output.clear_output(wait=True)
        with self.fit_output:
            print('Fit parameters:')
            for name, value in self.fit_parameters.items():
                print(f'  {name}: {value:g}')
        self._draw()

    def fit_values(self):
        """Return a copy of the fitted Y values, or None before fitting."""
        return None if self.fit is None else self.fit.copy()

    def _apply_axes(self, button=None):
        roles = [self.x_role.value, self.y_role.value]
        if len(set(roles)) != 2:
            raise ValueError('X and Y data must differ')
        indices = [['Original X', 'Original Y'].index(role) for role in roles]
        self.dataset.axes[0] = self._axis_values[indices[0]].copy()
        self.dataset.data = self._axis_values[indices[1]].copy()
        self.dataset.units = [self._axis_units[i] for i in indices]
        self.dataset.labels = [self.x_label.value, self.y_label.value]
        self.fit = None
        self.fit_parameters = {}
        self.fit_output.clear_output(wait=True)
        self._updating_cursor_slider = True
        try:
            self.cursor_slider.options = coordinate_slider_options(
                self.dataset.axes[0], self.dataset.units[0]
            )
            self.cursor_slider.value = tuple(self.cursors)
        finally:
            self._updating_cursor_slider = False
        self._draw()

    def _draw(self):
        x = self.dataset.axes[0]
        y = self.dataset.data
        self.axis.clear()
        self.axis.plot(x, y, color=self.data_color)
        for shown, index, color in ((self.show_c1.value, self.cursors[0], 'red'), (self.show_c2.value, self.cursors[1], 'green')):
            if shown:
                self.axis.plot(x[index], y[index], 'o', color=color)
                self.axis.axvline(x[index], color=color, alpha=0.5)
        if self.fit is not None and self.show_fit.value:
            self.axis.plot(x, self.fit, color=self.fit_color, zorder=10)
        self.axis.set(xlabel=self.x_label.value, ylabel=self.y_label.value, xscale='log' if self.log_x.value else 'linear', yscale='log' if self.log_y.value else 'linear')
        self.coordinates.value = f'C1: ({x[self.cursors[0]]:g}, {y[self.cursors[0]]:g}) &nbsp; C2: ({x[self.cursors[1]]:g}, {y[self.cursors[1]]:g})'
        self.figure.canvas.draw_idle()

    def profiles(self):
        lo, hi = sorted(self.cursors)
        return {'curve': self.dataset.data.copy(), 'between_cursors': self.dataset.data[lo:hi + 1].copy(), 'fit': None if self.fit is None else self.fit.copy()}

    def metadata(self):
        result = super().metadata()
        result['view'] = {'cursors': list(self.cursors), 'fit_function': self.fit_model.value, 'fit_parameters': copy.deepcopy(self.fit_parameters), 'data_color': self.data_color, 'fit_color': self.fit_color}
        return result

    def show(self):
        return _show_widget_canvas(self)

class ImageNotebookView(NotebookView):

    def __init__(self, dataset):
        super().__init__(dataset)
        if dataset.kind != 'image':
            raise ValueError('2D Image view requires 2D data')
        w, _ = _widgets()
        ny, nx = dataset.data.shape
        self.pos = [nx // 2, ny // 2]
        self.pos2 = [min(self.pos[0] + 5, nx - 1), self.pos[1]]
        self.reference = None
        self.selecting = False
        self._original = (dataset.data.copy(), [a.copy() for a in dataset.axes], list(dataset.labels), list(dataset.units))
        with plt.ioff():
            self.figure, self.axes = plt.subplots(2, 2, figsize=(9, 7))
        self.ax_x, self.ax_blank, self.ax_image, self.ax_y = self.axes.ravel()
        self.show_cursor = w.Checkbox(True, description='Show Cursor')
        self.show_second = w.Checkbox(True, description='Show Second Cursor')
        self.show_reference = w.Checkbox(True, description='Show Contrast Reference')
        self.x_slider = w.SelectionSlider(options=coordinate_slider_options(dataset.axes[0], dataset.units[0]), value=self.pos[0], description='X')
        self.y_slider = w.SelectionSlider(options=coordinate_slider_options(dataset.axes[1], dataset.units[1]), value=self.pos[1], description='Y')
        self.x2_slider = w.SelectionSlider(options=coordinate_slider_options(dataset.axes[0], dataset.units[0]), value=self.pos2[0], description='X2')
        self.y2_slider = w.SelectionSlider(options=coordinate_slider_options(dataset.axes[1], dataset.units[1]), value=self.pos2[1], description='Y2')
        self.cmap = w.Dropdown(options=sorted((name for name in plt.colormaps() if not name.endswith('_r'))), value='viridis', description='Colormap')
        self.invert = w.Checkbox(False, description='Invert')
        self.reference_button = w.Button(description='Select Contrast Box')
        self.ignore_reference = w.Checkbox(False, description='Ignore Contrast')
        self.x_axis = w.Dropdown(options=dataset.labels[:2], value=dataset.labels[0], description='X axis')
        self.y_axis = w.Dropdown(options=dataset.labels[:2], value=dataset.labels[1], description='Y axis')
        self.invert_x = w.Checkbox(False, description='Invert X')
        self.invert_y = w.Checkbox(False, description='Invert Y')
        self.apply_axes = w.Button(description='Apply Axes')
        self.controls = w.VBox([w.HBox([self.show_cursor, self.show_second, self.show_reference]), w.HBox([self.x_slider, self.y_slider, self.x2_slider, self.y2_slider]), w.HBox([self.reference_button, self.ignore_reference]), w.HBox([self.cmap, self.invert]), w.HBox([self.x_axis, self.y_axis, self.invert_x, self.invert_y, self.apply_axes])])
        for widget in (self.show_cursor, self.show_second, self.show_reference, self.x_slider, self.y_slider, self.x2_slider, self.y2_slider, self.cmap, self.invert, self.ignore_reference):
            widget.observe(self._changed, names='value')
        self.drag_cursor = False
        self._updating_sliders = False
        self.reference_button.on_click(lambda b: setattr(self, 'selecting', True))
        self.apply_axes.on_click(self._apply_axes)
        self.figure.canvas.mpl_connect('button_press_event', self._press)
        self.figure.canvas.mpl_connect('motion_notify_event', self._motion)
        self.figure.canvas.mpl_connect('button_release_event', self._release)
        self._draw()

    def _changed(self, change=None):
        if self._updating_sliders:
            return
        self.pos = [self.x_slider.value, self.y_slider.value]
        self.pos2 = [self.x2_slider.value, self.y2_slider.value]
        self._draw()

    def _point(self, event):
        x, y = self.dataset.axes
        return [int(np.argmin(abs(x - event.xdata))), int(np.argmin(abs(y - event.ydata)))]

    def _press(self, event):
        if event.inaxes != self.ax_image or event.xdata is None:
            return
        point = self._point(event)
        if self.selecting:
            self.reference = [point, point.copy()]
        else:
            distance2 = np.hypot(point[0] - self.pos2[0], point[1] - self.pos2[1])
            self.drag_cursor = 2 if self.show_second.value and distance2 < 2 else 1
            sliders = (self.x2_slider, self.y2_slider) if self.drag_cursor == 2 else (self.x_slider, self.y_slider)
            sliders[0].value, sliders[1].value = point

    def _motion(self, event):
        if self.selecting and self.reference and (event.inaxes == self.ax_image) and (event.xdata is not None):
            self.reference[1] = self._point(event)
            self._draw()
        elif self.drag_cursor and event.inaxes == self.ax_image and (event.xdata is not None):
            sliders = (self.x2_slider, self.y2_slider) if self.drag_cursor == 2 else (self.x_slider, self.y_slider)
            sliders[0].value, sliders[1].value = self._point(event)

    def _release(self, event):
        if self.selecting and event.inaxes == self.ax_image and (event.xdata is not None):
            self.reference[1] = self._point(event)
            self.selecting = False
            self._draw()
        self.drag_cursor = False

    def _apply_axes(self, button=None):
        data, axes, labels, units = self._original
        requested = [self.x_axis.value, self.y_axis.value]
        if len(set(requested)) != 2:
            raise ValueError('X and Y axes must differ')
        order = [labels.index(value) for value in requested]
        self.dataset.data = (data if order == [0, 1] else data.T).copy()
        self.dataset.axes = [axes[i].copy() for i in order]
        self.dataset.labels = [labels[i] for i in order] + labels[2:]
        self.dataset.units = [units[i] for i in order] + units[2:]
        for i, inverted in enumerate((self.invert_x.value, self.invert_y.value)):
            if inverted:
                self.dataset.axes[i] *= -1
        ny, nx = self.dataset.data.shape
        self.pos = [nx // 2, ny // 2]
        self.pos2 = [min(nx // 2 + 5, nx - 1), ny // 2]
        self._updating_sliders = True
        try:
            x_options = coordinate_slider_options(self.dataset.axes[0], self.dataset.units[0])
            y_options = coordinate_slider_options(self.dataset.axes[1], self.dataset.units[1])
            self.x_slider.options = self.x2_slider.options = x_options
            self.y_slider.options = self.y2_slider.options = y_options
            self.x_slider.value, self.y_slider.value = self.pos
            self.x2_slider.value, self.y2_slider.value = self.pos2
        finally:
            self._updating_sliders = False
        self.reference = None
        self._draw()

    def _draw(self):
        for ax in self.axes.ravel():
            ax.clear()
        data = self.dataset.data
        x, y = self.dataset.axes
        ix, iy = self.pos
        cmap = display_colormap(self.cmap.value, self.invert.value)
        shown = self.ax_image.imshow(data, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap)
        limits = image_contrast_limits(data, None if self.ignore_reference.value else self.reference)
        if limits:
            shown.set_clim(*limits)
        self.ax_x.plot(x, data[iy])
        self.ax_y.plot(data[:, ix], y)
        if self.show_second.value:
            distance, line = image_line_profile(self.dataset, self.pos, self.pos2)
            self.ax_blank.plot(distance, line)
        else:
            self.ax_blank.text(0.5, 0.5, 'Second cursor disabled', ha='center', transform=self.ax_blank.transAxes)
        if self.show_cursor.value:
            self.ax_image.axvline(x[ix], color='cyan')
            self.ax_image.axhline(y[iy], color='cyan')
            if self.show_second.value:
                jx, jy = self.pos2
                self.ax_image.plot([x[ix], x[jx]], [y[iy], y[jy]], 'r-o')
        if self.reference and self.show_reference.value:
            from matplotlib.patches import Rectangle
            (x0, y0), (x1, y1) = self.reference
            xa, xb = sorted((x[x0], x[x1]))
            ya, yb = sorted((y[y0], y[y1]))
            self.ax_image.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, color='yellow'))
        self.figure.canvas.draw_idle()

    def profiles(self):
        ix, iy = self.pos
        result = {'x': self.dataset.data[iy, :].copy(), 'y': self.dataset.data[:, ix].copy()}
        if self.show_second.value:
            distance, line = image_line_profile(self.dataset, self.pos, self.pos2)
            result.update(line_distance=distance.copy(), line=line.copy())
        return result

    def images(self):
        return {'display': self.dataset.data.copy()}

    def metadata(self):
        result = super().metadata()
        result['view'] = {'cursor': list(self.pos), 'second_cursor': list(self.pos2), 'show_second_cursor': self.show_second.value, 'contrast_reference': copy.deepcopy(self.reference), 'colormap': self.cmap.value, 'inverted': self.invert.value, 'ignore_contrast': self.ignore_reference.value}
        return result

    def save_image(self, path, name='display'):
        if name != 'display':
            raise KeyError("2D Image view provides only 'display'")
        save_display_image(path, self.dataset.data, display_colormap(self.cmap.value, self.invert.value))

    def show(self):
        return _show_widget_canvas(self)

class HyperspectralNotebookView(NotebookView):

    def __init__(self, dataset):
        super().__init__(dataset)
        if dataset.kind != 'hyperspectral':
            raise ValueError('Hyperspectral view requires 3D data')
        w, _ = _widgets()
        ny, nx, nz = dataset.data.shape
        self.pos = initial_hyperspectral_position(dataset.data.shape, dataset.axes[2])
        self.reference = None
        self.selecting = False
        self._original = (dataset.data.copy(), [a.copy() for a in dataset.axes], list(dataset.labels), list(dataset.units))
        with plt.ioff():
            self.figure = plt.figure(figsize=(10, 8), constrained_layout=True)
            outer = self.figure.add_gridspec(2, 2)
            left_top = outer[0, 0].subgridspec(2, 1, height_ratios=[1, 2])
            right_bottom = outer[1, 1].subgridspec(1, 2, width_ratios=[2, 1])
            self.ax_x_profile = self.figure.add_subplot(left_top[0])
            self.ax_xz = self.figure.add_subplot(left_top[1], sharex=self.ax_x_profile)
            self.ax_spectrum = self.figure.add_subplot(outer[0, 1])
            self.ax_image = self.figure.add_subplot(outer[1, 0], sharex=self.ax_x_profile)
            self.ax_yz = self.figure.add_subplot(right_bottom[0], sharey=self.ax_image)
            self.ax_y_profile = self.figure.add_subplot(right_bottom[1], sharey=self.ax_image)
        self.show_cursor = w.Checkbox(True, description='Show Cursor')
        self.show_profile = w.Checkbox(True, description='Show Profile Cursors')
        self.show_reference = w.Checkbox(True, description='Show Contrast Reference')
        self.x_slider = w.SelectionSlider(options=coordinate_slider_options(dataset.axes[0], dataset.units[0]), value=self.pos[0], description='X')
        self.y_slider = w.SelectionSlider(options=coordinate_slider_options(dataset.axes[1], dataset.units[1]), value=self.pos[1], description='Y')
        self.z_slider = w.SelectionSlider(options=coordinate_slider_options(dataset.axes[2], dataset.units[2]), value=self.pos[2], description='Spectrum')
        self.mode = w.Dropdown(options=['Z Slice', 'RGB Channels'], description='Display')
        self.combine = w.Dropdown(options=['Average', 'Maximum', 'Minimum'], value='Average', description='Value')
        self.reference_button = w.Button(description='Select Contrast Box')
        self.ignore_reference = w.Checkbox(False, description='Ignore Contrast')
        self.cmap = w.Dropdown(options=sorted((n for n in plt.colormaps() if not n.endswith('_r'))), value='viridis', description='Colormap')
        self.invert = w.Checkbox(False, description='Invert')
        self.ranges = {c: w.SelectionRangeSlider(options=coordinate_slider_options(dataset.axes[2], dataset.units[2]), value=(0, nz - 1), description=c) for c in 'RGB'}
        self.zeros = {c: w.Checkbox(c != 'R', description='Zero') for c in 'RGB'}
        self.channel_invert = {c: w.Checkbox(False, description='Invert') for c in 'RGB'}
        self.axis_controls = [w.Dropdown(options=dataset.labels, value=dataset.labels[i], description=f"{'XYZ'[i]} axis") for i in range(3)] + [w.Checkbox(False, description=f"Invert {'XYZ'[i]}") for i in range(3)]
        self.apply_axes = w.Button(description='Apply Axes')
        self.mode_line = w.VBox([w.HBox([self.cmap, self.invert])])
        self.controls = w.VBox([w.HBox([self.show_cursor, self.show_profile, self.show_reference]), w.HBox([self.x_slider, self.y_slider, self.z_slider]), w.HBox([self.mode, self.reference_button, self.ignore_reference, self.combine]), self.mode_line, w.HBox([*self.axis_controls, self.apply_axes])])
        for slider in (self.x_slider, self.y_slider, self.z_slider):
            slider.observe(self._coordinates_changed, names='value')
        self.mode.observe(self._mode_changed, names='value')
        display_controls = [
            self.show_cursor, self.show_profile, self.show_reference,
            self.ignore_reference, self.combine, self.cmap, self.invert,
            *self.ranges.values(), *self.zeros.values(),
            *self.channel_invert.values(),
        ]
        for widget in display_controls:
            widget.observe(self._display_changed, names='value')
        self.drag_mode = None
        self._updating_sliders = False
        self.reference_selector = None
        self.reference_button.on_click(self._start_contrast_selection)
        self.apply_axes.on_click(self._apply_axes)
        self.figure.canvas.mpl_connect('button_press_event', self._press)
        self.figure.canvas.mpl_connect('motion_notify_event', self._motion)
        self.figure.canvas.mpl_connect('button_release_event', self._release)
        self._mode_controls()
        self._draw()

    def _mode_controls(self):
        w, _ = _widgets()
        self.mode_line.children = (w.HBox([self.cmap, self.invert]),) if self.mode.value == 'Z Slice' else tuple((w.HBox([self.ranges[c], self.zeros[c], self.channel_invert[c]]) for c in 'RGB'))

    def _coordinates_changed(self, change=None):
        if self._updating_sliders:
            return
        self.pos = [self.x_slider.value, self.y_slider.value, self.z_slider.value]
        self._draw()

    def _mode_changed(self, change=None):
        self._mode_controls()
        self._draw()

    def _display_changed(self, change=None):
        if self._updating_sliders:
            return
        self._draw()

    def _point(self, event):
        x, y, _ = self.dataset.axes
        return [int(np.argmin(abs(x - event.xdata))), int(np.argmin(abs(y - event.ydata)))]

    def _set_position(self, *, ix=None, iy=None, iz=None):
        next_position = [
            self.pos[0] if ix is None else int(ix),
            self.pos[1] if iy is None else int(iy),
            self.pos[2] if iz is None else int(iz),
        ]
        if next_position == self.pos:
            return
        self.pos = next_position
        self._updating_sliders = True
        try:
            self.x_slider.value, self.y_slider.value, self.z_slider.value = self.pos
        finally:
            self._updating_sliders = False
        self._draw()

    def _start_contrast_selection(self, button=None):
        self.selecting = True
        self.drag_mode = None
        self.reference_selector.set_active(True)

    def _reset_reference_selector(self):
        if self.reference_selector is not None:
            self.reference_selector.disconnect_events()
        self.reference_selector = RectangleSelector(
            self.ax_image, self._contrast_selected,
            useblit=False, button=[1], interactive=True,
            minspanx=0, minspany=0, spancoords='data',
        )
        self.reference_selector.set_active(self.selecting)

    def _contrast_selected(self, press_event, release_event):
        if press_event.xdata is None or release_event.xdata is None:
            return
        self.reference = [self._point(press_event), self._point(release_event)]
        self.selecting = False
        self.reference_selector.set_active(False)
        self._draw()

    def _press(self, event):
        if event.xdata is None or self.selecting:
            return
        if event.inaxes == self.ax_image:
            self.drag_mode = 'xy'
            point = self._point(event)
            self._set_position(ix=point[0], iy=point[1])
        elif event.inaxes == self.ax_spectrum:
            self.drag_mode = 'z-spectrum'
            self._set_position(iz=nearest_axis_index(self.dataset.axes[2], event.xdata))
        elif event.inaxes == self.ax_xz and event.ydata is not None:
            self.drag_mode = 'z-xz'
            self._set_position(iz=nearest_axis_index(self.dataset.axes[2], event.ydata))
        elif event.inaxes == self.ax_yz:
            self.drag_mode = 'z-yz'
            self._set_position(iz=nearest_axis_index(self.dataset.axes[2], event.xdata))

    def _motion(self, event):
        if event.xdata is None or self.selecting:
            return
        if self.drag_mode == 'xy' and event.inaxes == self.ax_image:
            point = self._point(event)
            self._set_position(ix=point[0], iy=point[1])
        elif self.drag_mode == 'z-spectrum' and event.inaxes == self.ax_spectrum:
            self._set_position(iz=nearest_axis_index(self.dataset.axes[2], event.xdata))
        elif self.drag_mode == 'z-xz' and event.inaxes == self.ax_xz and event.ydata is not None:
            self._set_position(iz=nearest_axis_index(self.dataset.axes[2], event.ydata))
        elif self.drag_mode == 'z-yz' and event.inaxes == self.ax_yz:
            self._set_position(iz=nearest_axis_index(self.dataset.axes[2], event.xdata))

    def _release(self, event):
        self.drag_mode = None

    def _apply_axes(self, button=None):
        data, axes, labels, units = self._original
        requested = [widget.value for widget in self.axis_controls[:3]]
        if len(set(requested)) != 3:
            raise ValueError('X, Y, and Z axes must differ')
        source = [labels.index(value) for value in requested]
        dimension = {0: 1, 1: 0, 2: 2}
        permutation = [dimension[source[1]], dimension[source[0]], dimension[source[2]]]
        self.dataset.data = np.transpose(data, permutation).copy()
        self.dataset.axes = [axes[i].copy() for i in source]
        self.dataset.labels = [labels[i] for i in source]
        self.dataset.units = [units[i] for i in source]
        for i, widget in enumerate(self.axis_controls[3:]):
            if widget.value:
                self.dataset.axes[i] *= -1
        ny, nx, nz = self.dataset.data.shape
        self.pos = initial_hyperspectral_position(self.dataset.data.shape, self.dataset.axes[2])
        self._updating_sliders = True
        try:
            self.x_slider.options = coordinate_slider_options(self.dataset.axes[0], self.dataset.units[0])
            self.y_slider.options = coordinate_slider_options(self.dataset.axes[1], self.dataset.units[1])
            z_options = coordinate_slider_options(self.dataset.axes[2], self.dataset.units[2])
            self.z_slider.options = z_options
            for slider in self.ranges.values():
                slider.options = z_options
                slider.value = (0, nz - 1)
            self.x_slider.value, self.y_slider.value, self.z_slider.value = self.pos
        finally:
            self._updating_sliders = False
        self.reference = None
        self._draw()

    def _rgb(self):
        channels = []
        for c in 'RGB':
            if self.zeros[c].value:
                values = np.zeros(self.dataset.data.shape[:2])
            else:
                lo, hi = self.ranges[c].value
                values = collapse_spectral_range(self.dataset.data[:, :, lo:hi + 1], self.combine.value)
                values = normalize_display_channel(values, self.reference, not self.ignore_reference.value)
                if self.channel_invert[c].value:
                    values = 1 - values
            channels.append(values)
        return np.stack(channels, axis=2)

    def _draw(self):
        axes = (
            self.ax_x_profile, self.ax_xz, self.ax_spectrum,
            self.ax_image, self.ax_yz, self.ax_y_profile,
        )
        for ax in axes:
            ax.clear()
        self._reset_reference_selector()
        x, y, z = self.dataset.axes
        ix, iy, iz = self.pos
        s = hyperspectral_sections(self.dataset.data, ix, iy, iz)
        xz, yz, zlim = orient_spectral_planes(s, z)
        image = s['xy'] if self.mode.value == 'Z Slice' else self._rgb()
        cmap = display_colormap(self.cmap.value, self.invert.value)
        shown = self.ax_image.imshow(image, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap if image.ndim == 2 else None)
        self.ax_spectrum.plot(z, s['z_profile'])
        self.ax_x_profile.plot(x, s['x_profile'], color='red')
        self.ax_xz.imshow(xz, origin='lower', aspect='auto', extent=[x[0], x[-1], *zlim], cmap=cmap)
        self.ax_yz.imshow(yz, origin='lower', aspect='auto', extent=[*zlim, y[0], y[-1]], cmap=cmap)
        self.ax_y_profile.plot(s['y_profile'], y, color='red')
        limits = image_contrast_limits(s['xy'], None if self.ignore_reference.value else self.reference)
        if image.ndim == 2 and limits:
            shown.set_clim(*limits)
        if self.show_cursor.value:
            self.ax_image.axvline(x[ix], color='cyan')
            self.ax_image.axhline(y[iy], color='cyan')
        if self.show_profile.value:
            self.ax_xz.axhline(z[iz], color='cyan')
            self.ax_yz.axvline(z[iz], color='cyan')
        if self.mode.value == 'RGB Channels':
            for channel in 'RGB':
                if not self.zeros[channel].value:
                    low, high = self.ranges[channel].value
                    self.ax_spectrum.axvspan(
                        z[low], z[high], color=channel.lower(), alpha=0.15
                    )
        if self.reference and self.show_reference.value:
            from matplotlib.patches import Rectangle
            (x0, y0), (x1, y1) = self.reference
            xa, xb = sorted((x[x0], x[x1]))
            ya, yb = sorted((y[y0], y[y1]))
            self.ax_image.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, color='yellow'))
        self.ax_spectrum.axvline(z[iz], color='lime')
        self.ax_x_profile.tick_params(axis='x', labelbottom=False)
        self.ax_xz.tick_params(axis='x', labelbottom=False)
        self.ax_yz.tick_params(axis='y', labelleft=False)
        self.ax_y_profile.tick_params(axis='y', labelleft=False)
        self.figure.canvas.draw_idle()

    def profiles(self):
        ix, iy, iz = self.pos
        s = hyperspectral_sections(self.dataset.data, ix, iy, iz)
        return {k: s[k].copy() for k in ('x_profile', 'y_profile', 'z_profile')}

    def images(self):
        ix, iy, iz = self.pos
        s = hyperspectral_sections(self.dataset.data, ix, iy, iz)
        return {'display': (s['xy'] if self.mode.value == 'Z Slice' else self._rgb()).copy(), 'xy': s['xy'].copy(), 'xz': s['xz'].copy(), 'yz': s['yz'].copy()}

    def metadata(self):
        result = super().metadata()
        result['view'] = {'cursor': list(self.pos), 'display_mode': self.mode.value, 'range_value': self.combine.value, 'contrast_reference': copy.deepcopy(self.reference), 'colormap': self.cmap.value, 'inverted': self.invert.value, 'ignore_contrast': self.ignore_reference.value, 'rgb_ranges': {c: list(self.ranges[c].value) for c in 'RGB'}, 'rgb_zero': {c: self.zeros[c].value for c in 'RGB'}, 'rgb_inverted': {c: self.channel_invert[c].value for c in 'RGB'}}
        return result

    def save_image(self, path, name='display'):
        images = self.images()
        if name not in images:
            raise KeyError(f'Unknown image {name!r}; choose from {tuple(images)}')
        colormap = display_colormap(self.cmap.value, self.invert.value) if images[name].ndim == 2 else None
        save_display_image(path, images[name], colormap)

    def show(self):
        return _show_widget_canvas(self)

def open_view(data: DataSet | np.ndarray, **kwargs) -> NotebookView:
    """Create the appropriate notebook view from a dataset or ndarray."""
    dataset = data if isinstance(data, DataSet) else dataset_from_array(data, **kwargs)
    return {'curve': CurveNotebookView, 'image': ImageNotebookView, 'hyperspectral': HyperspectralNotebookView}[dataset.kind](dataset)
