# HyperScope

HyperScope is a proof-of-concept Matplotlib workbench made in ChatGPT Work which
can work with curves, 2D images, and hyperspectral images. HyperScope is not a
finished or production-supported application.

## License, responsibility, and contributions

HyperScope is released under the [Zero-Clause BSD license](LICENSE). Anyone is
welcome to use it, contribute improvements, create a fork, or rewrite it from
scratch.

This software is provided **as-is**, without warranties or guarantees. Users
are solely responsible for deciding whether and how to use it, using it in a
responsible manner, protecting and backing up their systems and data, and
accepting all risks arising from its use. The authors and contributors are not
responsible for data loss, system damage, or other consequences resulting from
use of the software.

Back up important files before using HyperScope, and run it from its own
folder. Only open files and sessions you trust.

## Requirements and interfaces

HyperScope requires Python, NumPy, Matplotlib, and SciPy. It offers two
interfaces:

- a standalone desktop application using Tkinter; and
- an optional Jupyter interface using JupyterLab, ipywidgets, and ipympl.

The desktop application does not require Jupyter or its related packages. The
shared data and Jupyter module, `hyperscope.py`, does not import Tkinter.

### macOS Python note

The version of Python or Tkinter supplied by an older macOS installation may
not work with the desktop application. If necessary, install a current version
of Python from the [official Python website](https://www.python.org/downloads/)
instead of using an older system-provided Python.

## Install

From Terminal on macOS, install the required dependencies with:

```bash
python3 -m pip install numpy matplotlib scipy
```

For the optional Jupyter interface, also install:

```bash
python3 -m pip install jupyterlab ipywidgets ipympl
```

On Linux, Tkinter may be provided as an operating-system package such as
`python3-tk`.

## Run the desktop application

Change into the extracted HyperScope folder and run:

```bash
cd /path/to/HyperScope
python3 hyperscope_desktop.py
```

On systems where the Python command is `python`, use:

```bash
python hyperscope_desktop.py
```

Click the import box at the top of the view manager to select one or more files
with the standard system file picker. No drag-and-drop package is required.

## Run in Jupyter

Start JupyterLab from the HyperScope folder:

```bash
jupyter lab
```

`hyperscope.py` provides the shared data API and notebook views without
importing Tkinter:

```python
from hyperscope import import_data, import_numpy, dataset_from_array

dataset = import_data("samples/ramanSpecRshift.txt")
data, axes, metadata = import_numpy("samples/ramanSpecRshift.txt")
image = dataset_from_array(my_2d_ndarray, title="My image")
```

Enable the widget backend and open a view:

```python
%matplotlib widget
from hyperscope import open_view

view = open_view(my_ndarray)
view.show()
result = view.extract()       # data, metadata, profiles, images
array = view.to_numpy()
scipy_input = view.to_scipy() # ndarray accepted by scipy.signal/ndimage
view.save_numpy("result.npz")
view.save_scipy("result.mat") # SciPy-written MAT file
view.save_image("display.png")
```

SciPy does not define a separate dense-array or image file format. The
`to_scipy()` method therefore returns a NumPy ndarray, which is SciPy's native
dense input type. `.npy` and `.npz` preserve NumPy arrays; `save_scipy()` writes
a compressed `.mat` bundle using `scipy.io`; PNG and JPEG save rendered images.
See `HyperScope_Demo.ipynb` for complete Curve, 2D Image, and Hyperspectral
examples.

## Import behavior

Text files are inspected and ranked as a curve, 2D image, or hyperspectral
image. The confirmation dialog shows the detected type, dimensions, confidence,
and reason and allows the user to override the result. Supported sample
structures include:

- comment-header two-column curves;
- one-column curves, using the sample index as the X axis;
- a selected numeric row imported as a curve;
- `XPoints`/`YPoints` coordinate image tables;
- coordinate-plus-spectrum hyperspectral tables; and
- declared hyperspectral images with `nx ny nz` followed by a spectral-axis
  row.

## Sessions

`.hscope` files save arrays, axes, metadata, titles, derivation hierarchy,
cursors, and view state in one versioned binary file. This proof of concept
intentionally rejects sessions from other schema versions rather than carrying
compatibility code.

**Known limitation:** after opening an existing desktop `.hscope` session, the
application may not insist that the session be saved before closing. Save the
session manually before exiting if you want to keep your changes.

Only open sessions you trust because the session format uses Python pickle.

## Tests and extension

`DataSet`, parsing, numerical fitting, and each view class are separate so
future revisions can add parsers or tools without rewriting the interface.

Run the complete test suite from the HyperScope folder with:

```bash
python3 -m pytest -q
```

or, on systems where the Python command is `python`:

```bash
python -m pytest -q
```
