#!/usr/bin/env python3
"""Tk desktop interface for HyperScope.

Requires NumPy, Matplotlib, SciPy, and a working Python Tk installation.
Jupyter and ipywidgets are not required.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from hyperscope import *


def grouped_colormap_button(parent, variable: tk.StringVar) -> ttk.Menubutton:
    """Build a categorized colormap menu shared by image views."""
    button = ttk.Menubutton(parent, textvariable=variable, width=18)
    menu = tk.Menu(button, tearoff=False)
    button.configure(menu=menu)
    for category, names in grouped_colormaps():
        category_menu = tk.Menu(menu, tearoff=False)
        for name in names:
            category_menu.add_command(
                label=name,
                command=lambda selected=name: variable.set(selected),
            )
        menu.add_cascade(label=category, menu=category_menu)
    button._colormap_menu = menu
    return button


class BaseView(ttk.Frame):
    def __init__(self, app, ds: DataSet):
        super().__init__(app.notebook)
        self.app, self.ds = app, ds
        self.has_shared_cursors = ds.kind != "curve"
        self.state = {"metadata": True}
        if self.has_shared_cursors:
            self.state["cursors"] = True
        self._view_information: dict[str, Any] = {}

        self.title_row = ttk.Frame(self)
        self.title_row.pack(fill="x")
        ttk.Label(self.title_row, text="Title:").pack(side="left")
        self.title_var = tk.StringVar(value=ds.title)
        title_entry = ttk.Entry(
            self.title_row, textvariable=self.title_var, width=42
        )
        title_entry.pack(side="left", padx=3)
        self.title_var.trace_add("write", self._title_changed)
        self.subtitle_var = tk.StringVar(value=self.subtitle())
        ttk.Label(
            self.title_row, textvariable=self.subtitle_var, foreground="#666"
        ).pack(side="left", padx=8)

        self.new_view_bar = ttk.Frame(self)
        self.new_view_bar.pack(fill="x")
        ttk.Label(self.new_view_bar, text="To New View:").pack(side="left")
        ttk.Button(
            self.new_view_bar, text="Delete View", command=self.delete_view
        ).pack(side="right")
        ttk.Button(
            self.new_view_bar, text="Duplicate View", command=self.duplicate_view
        ).pack(side="right")

        self.export_bar = ttk.Frame(self)
        self.export_bar.pack(fill="x")
        ttk.Label(self.export_bar, text="Export:").pack(side="left")
        self.export_plot_button = ttk.Button(
            self.export_bar,
            text="Export Plot to File",
            command=self.export_view,
        )
        self.export_plot_button.pack(side="right")
        self.export_data_button = ttk.Button(
            self.export_bar, text="Export Data to File", command=self.export_data
        )
        self.export_data_button.pack(side="right")
        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True)
        self.plotframe = ttk.Frame(body)
        body.add(self.plotframe, weight=5)
        self.side = tk.Text(body, width=30, wrap="word")
        body.add(self.side, weight=1)
        self.fig = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self.plotframe)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # First options row is consistent across every view.
        self.view_controls = ttk.Frame(self)
        self.view_controls.pack(fill="x")
        self.metadata_var = tk.BooleanVar(value=True)
        self.cursors_var = (
            tk.BooleanVar(value=True) if self.has_shared_cursors else None
        )
        ttk.Checkbutton(
            self.view_controls, text="Show Metadata",
            variable=self.metadata_var, command=self._metadata_checkbox_changed
        ).pack(side="left")
        if self.has_shared_cursors:
            ttk.Checkbutton(
                self.view_controls, text="Show Cursors",
                variable=self.cursors_var, command=self._cursors_checkbox_changed
            ).pack(side="left")

        # Following rows belong to the specific view type.
        self.bottom = ttk.Frame(self)
        self.bottom.pack(fill="x")
        self.refresh_meta()

    def subtitle(self):
        if self.ds.parent:
            parent = self.app.datasets.get(self.ds.parent)
            source_title = parent.title if parent else self.ds.source
            return f"Derived from {source_title}: {self.ds.derivation}"
        return f"Source: {Path(self.ds.source).name}"

    def _title_changed(self, *_):
        self.ds.title = self.title_var.get()
        self.app.refresh_tree()
        for view in self.app.views.values():
            if view.ds.parent == self.ds.uid:
                view.refresh_subtitle()

    def refresh_subtitle(self):
        self.subtitle_var.set(self.subtitle())

    def duplicate_view(self):
        self.app.open_dataset(duplicate_dataset(self.ds))

    def delete_view(self):
        if messagebox.askyesno(
            "Delete View", f"Delete the view “{self.ds.title}”?"
        ):
            self.app.delete_view(self.ds.uid)

    def refresh_meta(self, extra=None):
        if extra is not None:
            self._view_information = dict(extra)
        self.side.delete("1.0", "end")
        self.side.insert(
            "end",
            metadata_sidebar_text(
                self.ds, self._view_information, self.metadata_var.get()
            ),
        )

    def _metadata_checkbox_changed(self):
        self.state["metadata"] = self.metadata_var.get()
        self.refresh_meta()

    def _cursors_checkbox_changed(self):
        self.state["cursors"] = self.cursors_var.get()
        self.draw()

    def export_view(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=PLOT_EXPORT_FILETYPES
        )
        if path:
            self.fig.savefig(path)

    def export_data(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=data_export_filetypes(self.ds.kind),
        )
        if not path:
            return
        suffix = Path(path).suffix.lower()
        allowed = {
            extension
            for _, patterns in data_export_filetypes(self.ds.kind)
            for pattern in patterns.split()
            for extension in [pattern.removeprefix("*")]
        }
        if suffix not in allowed:
            return messagebox.showerror(
                "Export data",
                f"{suffix or 'That file type'} is not available for "
                f"{self.ds.kind} data.",
            )
        if suffix in (".png", ".jpg", ".jpeg"):
            import matplotlib.pyplot as plt

            plt.imsave(
                path, raster_export_array(self.ds), cmap="viridis"
            )
        else:
            export_dataset_text(self.ds, path)

    def serial_state(self):
        return dict(self.state)

    def restore_state(self, state):
        self.state.update(state)
        self.metadata_var.set(self.state["metadata"])
        if self.has_shared_cursors:
            self.cursors_var.set(self.state["cursors"])
        self.draw()


class CurveView(BaseView):
    def __init__(self, app, ds):
        super().__init__(app, ds)
        self.ax = self.fig.add_subplot(111)
        self.canvas.get_tk_widget().configure(height=420)
        self.cursors = [len(ds.data) // 3, 2 * len(ds.data) // 3]
        self.drag = None
        self.fit_line = None
        self.style = default_curve_style()
        self.fit_style = default_fit_style()
        self.fit_parameters_result: dict[str, float] | None = None
        self.cursor_visible_vars = [
            tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        ]
        self.vertical_cursors_var = tk.BooleanVar(value=True)
        for index, color in enumerate(("red", "green")):
            ttk.Checkbutton(
                self.view_controls,
                text=f"Show Cursor {index + 1} ({color})",
                variable=self.cursor_visible_vars[index],
                command=self.draw,
            ).pack(side="left")
        ttk.Checkbutton(
            self.view_controls,
            text="Show Vertical Cursors",
            variable=self.vertical_cursors_var,
            command=self.draw,
        ).pack(side="left")
        self.show_fit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.view_controls, text="Show Fit",
            variable=self.show_fit_var, command=self.draw,
        ).pack(side="left")
        ttk.Button(
            self.new_view_bar,
            text="Range Within Cursors to New View",
            command=self.export_cursor_range,
        ).pack(side="right")
        ttk.Button(
            self.new_view_bar,
            text="Send X Axis to Other View",
            command=self.send_x,
        ).pack(side="right")

        ttk.Label(self.bottom, text="Data:").pack(side="left", padx=(4, 1))
        self.line_style_var = tk.StringVar(value="-")
        self.line_width_var = tk.StringVar(value="1.5")
        self.line_color_var = tk.StringVar(value="#1565c0")
        self.marker_var = tk.StringVar(value="None")
        self.marker_size_var = tk.StringVar(value="6")
        self.marker_color_var = tk.StringVar(value="#1565c0")
        display_controls = (
            ("Line style", self.line_style_var, CURVE_LINE_STYLES, 6),
            ("Line width", self.line_width_var, CURVE_LINE_WIDTHS, 5),
            ("Line color", self.line_color_var, CURVE_COLORS, 12),
            ("Marker", self.marker_var, CURVE_MARKERS, 6),
            ("Marker size", self.marker_size_var, CURVE_MARKER_SIZES, 5),
            ("Marker color", self.marker_color_var, CURVE_COLORS, 12),
        )
        for label, variable, values, width in display_controls:
            ttk.Label(self.bottom, text=label + ":").pack(
                side="left", padx=(4, 1)
            )
            box = ttk.Combobox(
                self.bottom, textvariable=variable, values=values,
                width=width, state="readonly",
            )
            box.pack(side="left")
            box.bind("<<ComboboxSelected>>", self.display_changed)

        self.fit_style_controls = ttk.Frame(self)
        self.fit_style_controls.pack(fill="x", after=self.bottom)
        ttk.Label(
            self.fit_style_controls, text="Fit Style:"
        ).pack(side="left", padx=(4, 1))
        self.fit_line_style_var = tk.StringVar(value="-")
        self.fit_line_width_var = tk.StringVar(value="2")
        self.fit_line_color_var = tk.StringVar(value="orange")
        self.fit_marker_var = tk.StringVar(value="None")
        self.fit_marker_size_var = tk.StringVar(value="6")
        self.fit_marker_color_var = tk.StringVar(value="orange")
        fit_display_controls = (
            ("Line style", self.fit_line_style_var, CURVE_LINE_STYLES, 6),
            ("Line width", self.fit_line_width_var, CURVE_LINE_WIDTHS, 5),
            ("Line color", self.fit_line_color_var, CURVE_COLORS, 12),
            ("Marker", self.fit_marker_var, CURVE_MARKERS, 6),
            ("Marker size", self.fit_marker_size_var, CURVE_MARKER_SIZES, 5),
            ("Marker color", self.fit_marker_color_var, CURVE_COLORS, 12),
        )
        for label, variable, values, width in fit_display_controls:
            ttk.Label(
                self.fit_style_controls, text=label + ":"
            ).pack(side="left", padx=(4, 1))
            box = ttk.Combobox(
                self.fit_style_controls, textvariable=variable, values=values,
                width=width, state="readonly",
            )
            box.pack(side="left")
            box.bind("<<ComboboxSelected>>", self.fit_style_changed)

        self.fit_controls = ttk.Frame(self)
        self.fit_controls.pack(fill="x", after=self.fit_style_controls)
        ttk.Label(
            self.fit_controls, text="Fit:"
        ).pack(side="left", padx=(4, 1))
        self.fit_model_var = tk.StringVar(value="gaussian")
        self.fit_model_box = ttk.Combobox(
            self.fit_controls, textvariable=self.fit_model_var,
            values=CURVE_FIT_FUNCTIONS, width=15, state="readonly",
        )
        self.fit_model_box.pack(side="left")
        self.fit_model_box.bind("<<ComboboxSelected>>", self.fit_model_changed)
        ttk.Label(
            self.fit_controls, text="Parameters:"
        ).pack(side="left", padx=(6, 1))
        self.fit_parameters_var = tk.StringVar(value="")
        self.fit_parameters_entry = ttk.Entry(
            self.fit_controls, textvariable=self.fit_parameters_var, width=16
        )
        self.fit_parameters_entry.pack(side="left")
        self.fit_parameters_entry.configure(state="disabled")
        self.fit_between_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.fit_controls, text="Between Markers",
            variable=self.fit_between_var,
        ).pack(side="left", padx=4)
        ttk.Button(
            self.fit_controls, text="Fit", command=self.fit_from_controls
        ).pack(side="left")

        self.axis_controls = ttk.Frame(self)
        self.axis_controls.pack(fill="x", after=self.fit_controls)
        ttk.Label(
            self.axis_controls, text="Axes:"
        ).pack(side="left", padx=(4, 1))
        self.axis_role_names = ["Original X", "Original Y"]
        self.axis_source_values = [
            ds.axes[0].copy(), ds.data.copy()
        ]
        self.axis_source_units = list(ds.units[:2])
        self.axis_vars = [
            tk.StringVar(value=self.axis_role_names[0]),
            tk.StringVar(value=self.axis_role_names[1]),
        ]
        self.axis_label_vars = [
            tk.StringVar(value=ds.labels[0]),
            tk.StringVar(value=ds.labels[1]),
        ]
        self.log_axis_vars = [
            tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        ]
        for index, role in enumerate(("X axis", "Y axis")):
            ttk.Label(
                self.axis_controls, text=role + ":"
            ).pack(side="left", padx=(4, 1))
            ttk.Combobox(
                self.axis_controls, textvariable=self.axis_vars[index],
                values=self.axis_role_names, width=10, state="readonly",
            ).pack(side="left")
            ttk.Label(
                self.axis_controls, text="Label:"
            ).pack(side="left", padx=(3, 1))
            ttk.Entry(
                self.axis_controls,
                textvariable=self.axis_label_vars[index],
                width=10,
            ).pack(side="left")
            ttk.Checkbutton(
                self.axis_controls, text=f"Log {role[0]}",
                variable=self.log_axis_vars[index],
                command=self.axis_scale_changed,
            ).pack(side="left")
        ttk.Button(
            self.axis_controls,
            text="Apply Axes / Labels",
            command=self.apply_curve_axes,
        ).pack(side="left")
        ttk.Button(
            self.axis_controls, text="X Ticks…",
            command=lambda: self.tick_dialog("x"),
        ).pack(side="left", padx=(4, 1))
        ttk.Button(
            self.axis_controls, text="Y Ticks…",
            command=lambda: self.tick_dialog("y"),
        ).pack(side="left")
        self.canvas.mpl_connect("button_press_event", self.press)
        self.canvas.mpl_connect("motion_notify_event", self.motion)
        self.canvas.mpl_connect("button_release_event", lambda e:setattr(self,"drag",None))
        self.draw()
    def draw(self):
        self.ax.clear()
        x = self.ds.axes[0]
        y = self.ds.data
        self.ax.set_xscale(
            curve_axis_scale(x, self.log_axis_vars[0].get())
        )
        self.ax.set_yscale(
            curve_axis_scale(y, self.log_axis_vars[1].get())
        )
        self.ax.plot(x, y, **self.style)
        self.ax.set(
            xlabel=f"{self.ds.labels[0]} ({self.ds.units[0]})",
            ylabel=f"{self.ds.labels[1]} ({self.ds.units[1]})",
            title=self.ds.title,
        )
        if self.fit_line is not None and self.show_fit_var.get():
            self.ax.plot(
                x, self.fit_line, label="fit", zorder=10, **self.fit_style
            )
            self.ax.legend()
        cursor_visibility = [
            variable.get() for variable in self.cursor_visible_vars
        ]
        for index, cursor in enumerate(self.cursors):
            if cursor_visibility[index]:
                self.ax.plot(
                    float(x[cursor]), float(y[cursor]), marker="o",
                    color=CURVE_CURSOR_COLORS[index], linestyle="none",
                )
            if (
                cursor_visibility[index]
                and self.vertical_cursors_var.get()
            ):
                self.ax.axvline(
                    x[cursor], color=CURVE_CURSOR_COLORS[index], alpha=.8
                )
        information = curve_sidebar_information(
            x, y, self.cursors, cursor_visibility
        )
        information.update(
            fit_sidebar_information(
                self.show_fit_var.get(), self.fit_model_var.get(),
                self.fit_line, self.fit_parameters_result,
            )
        )
        self.refresh_meta(information)
        self.canvas.draw_idle()

    def export_cursor_range(self):
        self.app.open_dataset(curve_range_dataset(self.ds, self.cursors))

    def press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x = self.ds.axes[0]
        index = int(np.argmin(abs(x - event.xdata)))
        visible = [
            index for index, variable in enumerate(self.cursor_visible_vars)
            if variable.get()
        ]
        if not visible:
            return
        self.drag = min(
            visible,
            key=lambda cursor: abs(index - self.cursors[cursor]),
        )
        self.cursors[self.drag] = index
        self.draw()

    def motion(self, event):
        if (
            self.drag is not None
            and event.inaxes == self.ax
            and event.xdata is not None
        ):
            self.cursors[self.drag] = int(
                np.argmin(abs(self.ds.axes[0] - event.xdata))
            )
            self.draw()
    def display_changed(self, _event=None):
        try:
            self.style = curve_style_from_controls(
                self.line_style_var.get(), self.line_width_var.get(),
                self.line_color_var.get(), self.marker_var.get(),
                self.marker_size_var.get(), self.marker_color_var.get(),
            )
            self.draw()
        except ValueError as error:
            messagebox.showerror("Display options", str(error))
    def fit_style_changed(self, _event=None):
        try:
            self.fit_style = curve_style_from_controls(
                self.fit_line_style_var.get(), self.fit_line_width_var.get(),
                self.fit_line_color_var.get(), self.fit_marker_var.get(),
                self.fit_marker_size_var.get(), self.fit_marker_color_var.get(),
            )
            self.draw()
        except ValueError as error:
            messagebox.showerror("Fit style", str(error))
    def fit_model_changed(self, _event=None):
        polynomial = self.fit_model_var.get() == "polynomial"
        self.fit_parameters_entry.configure(
            state="normal" if polynomial else "disabled"
        )
        if polynomial and not self.fit_parameters_var.get().strip():
            self.fit_parameters_var.set("order=3")
        elif not polynomial:
            self.fit_parameters_var.set("")
    def fit_from_controls(self):
        try:
            self.fit_line, parameters = fit_curve_selection(
                self.ds.axes[0], self.ds.data, self.fit_model_var.get(),
                self.fit_parameters_var.get(), self.fit_between_var.get(),
                self.cursors,
            )
            self.fit_parameters_result = labeled_fit_parameters(
                self.fit_model_var.get(), parameters
            )
            self.ds.metadata["Last fit"] = (
                f"{self.fit_model_var.get()}: {parameters}"
            )
            self.draw()
        except Exception as error:
            messagebox.showerror("Fit failed", str(error))
    def apply_curve_axes(self):
        requested = [variable.get() for variable in self.axis_vars]
        if len(set(requested)) != 2:
            return messagebox.showerror(
                "Axes", "X and Y must use different data."
            )
        indices = [self.axis_role_names.index(value) for value in requested]
        self.ds.axes[0] = self.axis_source_values[indices[0]].copy()
        self.ds.data = self.axis_source_values[indices[1]].copy()
        self.ds.units = [
            self.axis_source_units[indices[0]],
            self.axis_source_units[indices[1]],
        ]
        self.ds.labels = [
            variable.get() for variable in self.axis_label_vars
        ]
        self.fit_line = None
        self.axis_scale_changed()
    def axis_scale_changed(self):
        errors = []
        for values, variable in zip(
            (self.ds.axes[0], self.ds.data), self.log_axis_vars
        ):
            try:
                curve_axis_scale(values, variable.get())
            except ValueError as error:
                variable.set(False)
                errors.append(str(error))
        if errors:
            messagebox.showerror("Log axes", errors[0])
        self.draw()
    def tick_dialog(self, axis_name):
        locations = simpledialog.askstring(
            f"{axis_name.upper()} ticks",
            "Optional comma-separated tick locations (blank = automatic)",
        )
        labels = simpledialog.askstring(
            f"{axis_name.upper()} tick labels",
            "Optional comma-separated labels",
        )
        try:
            if locations:
                ticks = [float(value) for value in locations.split(",")]
                tick_labels = labels.split(",") if labels else None
                if axis_name == "x":
                    self.ax.set_xticks(ticks, tick_labels)
                else:
                    self.ax.set_yticks(ticks, tick_labels)
            self.canvas.draw_idle()
        except ValueError:
            messagebox.showerror("Ticks", "Tick locations must be numbers.")
    def send_x(self):
        choices = [
            view for view in self.app.views.values()
            if view is not self
            and view.ds.kind in {"curve", "hyperspectral"}
        ]
        if not choices:
            return messagebox.showinfo(
                "Send X Axis", "No other Curve or hyperspectral view is open."
            )
        names = "\n".join(
            f"{index}: {view.ds.title} ({view.ds.kind})"
            for index, view in enumerate(choices)
        )
        index = simpledialog.askinteger(
            "Send X Axis to Other View",
            "Choose the target view:\n\n" + names,
            initialvalue=0, minvalue=0, maxvalue=len(choices) - 1,
        )
        if index is None:
            return
        try:
            derived = dataset_with_replaced_axis(
                self.ds, choices[index].ds
            )
        except ValueError as error:
            return messagebox.showerror("Send X Axis", str(error))
        self.app.open_dataset(derived)
    def serial_state(self):
        return {
            **self.state,
            "cursors": self.cursors,
            "cursor_visibility": [
                variable.get() for variable in self.cursor_visible_vars
            ],
            "vertical_cursors": self.vertical_cursors_var.get(),
            "show_fit": self.show_fit_var.get(),
            "style": self.style,
            "fit_style": self.fit_style,
            "fit_line": self.fit_line,
            "fit_parameters_result": self.fit_parameters_result,
            "fit_model": self.fit_model_var.get(),
            "fit_parameters": self.fit_parameters_var.get(),
            "fit_between": self.fit_between_var.get(),
            "axis_roles": [variable.get() for variable in self.axis_vars],
            "axis_labels": [
                variable.get() for variable in self.axis_label_vars
            ],
            "log_axes": [variable.get() for variable in self.log_axis_vars],
            "axis_source_values": self.axis_source_values,
            "axis_source_units": self.axis_source_units,
        }
    def restore_state(self,s):
        self.cursors = list(s.pop("cursors"))
        for variable, visible in zip(
            self.cursor_visible_vars, s.pop("cursor_visibility")
        ):
            variable.set(visible)
        self.vertical_cursors_var.set(s.pop("vertical_cursors"))
        self.show_fit_var.set(s.pop("show_fit"))
        self.style = s.pop("style")
        self.line_style_var.set(self.style["linestyle"] or "None")
        self.line_width_var.set(str(self.style["linewidth"]))
        self.line_color_var.set(self.style["color"])
        self.marker_var.set(self.style["marker"] or "None")
        self.marker_size_var.set(str(self.style["markersize"]))
        self.marker_color_var.set(self.style["markerfacecolor"])
        self.fit_style = s.pop("fit_style")
        self.fit_line_style_var.set(self.fit_style["linestyle"] or "None")
        self.fit_line_width_var.set(str(self.fit_style["linewidth"]))
        self.fit_line_color_var.set(self.fit_style["color"])
        self.fit_marker_var.set(self.fit_style["marker"] or "None")
        self.fit_marker_size_var.set(str(self.fit_style["markersize"]))
        self.fit_marker_color_var.set(self.fit_style["markerfacecolor"])
        self.fit_line = s.pop("fit_line")
        self.fit_parameters_result = s.pop("fit_parameters_result")
        self.fit_model_var.set(s.pop("fit_model"))
        self.fit_parameters_var.set(s.pop("fit_parameters"))
        self.fit_model_changed()
        self.fit_between_var.set(s.pop("fit_between"))
        for variable, value in zip(self.axis_vars, s.pop("axis_roles")):
            variable.set(value)
        for variable, value in zip(
            self.axis_label_vars, s.pop("axis_labels")
        ):
            variable.set(value)
        for variable, value in zip(
            self.log_axis_vars, s.pop("log_axes")
        ):
            variable.set(value)
        self.axis_source_values = s.pop("axis_source_values")
        self.axis_source_units = s.pop("axis_source_units")
        super().restore_state(s)


class ImageView(BaseView):
    def __init__(self, app, ds):
        super().__init__(app, ds)
        ny, nx = ds.data.shape
        self.p1 = [nx // 2, ny // 2]
        self.p2 = [min(self.p1[0] + 5, nx - 1), self.p1[1]]
        self.drag = None
        self.reference = None
        self.shown_instructions: set[str] = set()
        self.second_var = tk.BooleanVar(value=True)
        self.cmap = tk.StringVar(value="viridis")
        self.invert_colors = tk.BooleanVar(value=False)
        self.show_reference = tk.BooleanVar(value=True)
        self.ignore_reference = tk.BooleanVar(value=False)
        self.axis_ids = list(ds.labels[:2])
        self.axis_vars = [tk.StringVar(value=value) for value in self.axis_ids]
        self.invert_vars = [tk.BooleanVar(value=False) for _ in range(2)]
        self.applied_inversions = [False, False]

        ttk.Checkbutton(
            self.view_controls, text="Show Second Cursor",
            variable=self.second_var, command=self.draw
        ).pack(side="left")
        ttk.Checkbutton(
            self.view_controls, text="Show Contrast Reference",
            variable=self.show_reference, command=self.draw
        ).pack(side="left", padx=4)

        grid = self.fig.add_gridspec(2, 2)
        self.ax_im = self.fig.add_subplot(grid[1, 0])
        self.ax_x = self.fig.add_subplot(grid[0, 0], sharex=self.ax_im)
        self.ax_line = self.fig.add_subplot(grid[0, 1])
        self.ax_y = self.fig.add_subplot(grid[1, 1], sharey=self.ax_im)

        ttk.Label(self.bottom, text="Colormap:").pack(side="left", padx=(4, 1))
        self.cmap_button = grouped_colormap_button(
            self.bottom, self.cmap
        )
        self.cmap_button.pack(side="left")
        ttk.Checkbutton(
            self.bottom, text="Invert Colors",
            variable=self.invert_colors, command=self.draw
        ).pack(side="left", padx=4)
        ttk.Button(
            self.bottom, text="Contrast Reference", command=self.reference_mode
        ).pack(side="left", padx=4)
        ttk.Checkbutton(
            self.bottom, text="Ignore Contrast Reference",
            variable=self.ignore_reference, command=self.draw
        ).pack(side="left", padx=4)
        self.axis_controls = ttk.Frame(self)
        self.axis_controls.pack(fill="x", after=self.bottom)
        for index, role in enumerate(("X axis", "Y axis")):
            ttk.Label(
                self.axis_controls, text=role + ":"
            ).pack(side="left", padx=(5, 1))
            ttk.Combobox(
                self.axis_controls, textvariable=self.axis_vars[index],
                values=self.axis_ids, width=11, state="readonly"
            ).pack(side="left")
            ttk.Checkbutton(
                self.axis_controls, text="Invert",
                variable=self.invert_vars[index]
            ).pack(side="left")
        ttk.Button(
            self.axis_controls, text="Apply axes", command=self.apply_axes
        ).pack(side="left", padx=4)
        ttk.Button(
            self.new_view_bar, text="Line Profile to New View",
            command=self.export_line
        ).pack(side="right")
        ttk.Button(
            self.new_view_bar, text="Y Profile to New View",
            command=lambda: self.export_profile("y")
        ).pack(side="right")
        ttk.Button(
            self.new_view_bar, text="X Profile to New View",
            command=lambda: self.export_profile("x")
        ).pack(side="right")
        ttk.Button(
            self.export_bar, text="Export Image to File",
            command=self.export_image_file,
        ).pack(side="right", before=self.export_data_button)

        self.canvas.mpl_connect("button_press_event", self.press)
        self.canvas.mpl_connect("motion_notify_event", self.motion)
        self.canvas.mpl_connect("button_release_event", self.release)
        self.cmap.trace_add("write", lambda *_: self.draw())
        self.title_var.trace_add("write", lambda *_: self.draw())
        self.draw()

    @property
    def second(self):
        return self.second_var.get()

    def draw(self):
        data = self.ds.data
        x, y = self.ds.axes
        ix, iy = self.p1
        for axis in (self.ax_x, self.ax_line, self.ax_im, self.ax_y):
            axis.clear()
        shown = self.ax_im.imshow(
            data, origin="lower", aspect="auto",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap=display_colormap(self.cmap.get(), self.invert_colors.get())
        )
        limits = image_contrast_limits(
            data, None if self.ignore_reference.get() else self.reference
        )
        if limits:
            shown.set_clim(*limits)
        if self.reference and self.show_reference.get():
            from matplotlib.patches import Rectangle
            (x0, y0), (x1, y1) = self.reference
            xa, xb = sorted((x[x0], x[x1]))
            ya, yb = sorted((y[y0], y[y1]))
            self.ax_im.add_patch(Rectangle(
                (xa, ya), xb-xa, yb-ya,
                fill=False, color="yellow", lw=1.5
            ))
        self.ax_x.plot(x, data[iy])
        self.ax_y.plot(data[:, ix], y)
        if self.state["cursors"]:
            self.ax_im.axvline(x[ix], color="cyan")
            self.ax_im.axhline(y[iy], color="cyan")
            if self.second:
                jx, jy = self.p2
                self.ax_im.plot([x[ix], x[jx]], [y[iy], y[jy]], "r-o")
        if self.second:
            distance, values = image_line_profile(
                self.ds, self.p1, self.p2
            )
            self.ax_line.plot(distance, values)
        else:
            self.ax_line.text(
                .5, .5, "Second cursor disabled",
                ha="center", transform=self.ax_line.transAxes
            )
        self.ax_im.set(
            xlabel=f"{self.ds.labels[0]} ({self.ds.units[0]})",
            ylabel=f"{self.ds.labels[1]} ({self.ds.units[1]})"
        )
        self.ax_x.tick_params(axis="x", labelbottom=False)
        self.ax_y.tick_params(axis="y", labelleft=False)
        self.fig.suptitle(self.ds.title)
        self.refresh_meta({
            "Cursor 1": (x[ix], y[iy]),
            "Cursor 2": (
                (x[self.p2[0]], y[self.p2[1]]) if self.second else "off"
            ),
            "Contrast reference": self.reference,
            "Contrast reference shown": self.show_reference.get(),
            "Contrast reference ignored": self.ignore_reference.get(),
        })
        self.canvas.draw_idle()

    def point(self, event):
        x, y = self.ds.axes
        return [
            int(np.argmin(abs(x-event.xdata))),
            int(np.argmin(abs(y-event.ydata)))
        ]

    def press(self, event):
        if event.inaxes != self.ax_im or event.xdata is None:
            return
        point = self.point(event)
        if self.drag == "reference-ready":
            self.reference = [point, point.copy()]
            self.drag = "reference"
        else:
            distances = [
                np.hypot(point[0]-self.p1[0], point[1]-self.p1[1]),
                np.hypot(point[0]-self.p2[0], point[1]-self.p2[1])
            ]
            self.drag = 1 if self.second and distances[1] < 2 else 0
            (self.p2 if self.drag else self.p1)[:] = point
        self.draw()

    def motion(self, event):
        if event.inaxes != self.ax_im or event.xdata is None:
            return
        if self.drag == "reference":
            self.reference[1] = self.point(event)
        elif self.drag in (0, 1):
            (self.p2 if self.drag else self.p1)[:] = self.point(event)
        else:
            return
        self.draw()

    def release(self, _event):
        self.drag = None

    def reference_mode(self):
        self.drag = "reference-ready"
        if consume_first_use(self.shown_instructions, "contrast_reference"):
            messagebox.showinfo(
                "Contrast Reference",
                "Drag a rectangle on the image. Select this tool again to replace it."
            )

    def apply_axes(self):
        requested = [variable.get() for variable in self.axis_vars]
        if len(set(requested)) != 2:
            messagebox.showerror("Axes", "X and Y must use different source axes.")
            return
        current = list(self.axis_ids)
        source_indices = [current.index(name) for name in requested]
        axis_to_data_dimension = {0: 1, 1: 0}
        permutation = [
            axis_to_data_dimension[source_indices[1]],
            axis_to_data_dimension[source_indices[0]],
        ]
        self.ds.data = np.transpose(self.ds.data, permutation)
        self.ds.axes = [self.ds.axes[index] for index in source_indices]
        self.ds.labels = (
            [self.ds.labels[index] for index in source_indices]
            + self.ds.labels[2:]
        )
        self.ds.units = (
            [self.ds.units[index] for index in source_indices]
            + self.ds.units[2:]
        )
        self.axis_ids = requested
        for index in range(2):
            desired = self.invert_vars[index].get()
            was_inverted = self.applied_inversions[source_indices[index]]
            if desired != was_inverted:
                self.ds.axes[index] = -self.ds.axes[index]
        self.applied_inversions = [variable.get() for variable in self.invert_vars]
        ny, nx = self.ds.data.shape
        self.p1 = [nx // 2, ny // 2]
        self.p2 = [min(self.p1[0] + 5, nx - 1), self.p1[1]]
        self.reference = None
        self.draw()

    def export_line(self):
        if not self.second:
            return
        distance, values = image_line_profile(
            self.ds, self.p1, self.p2
        )
        derived = DataSet(
            "curve", values, [distance],
            f"{self.ds.title} line", self.ds.source, {},
            ["distance", "a.u."], ["Position", "Signal"],
            parent=self.ds.uid, derivation={"p1": self.p1, "p2": self.p2}
        )
        self.app.open_dataset(derived)

    def export_profile(self, axis):
        self.app.open_dataset(
            image_profile_dataset(
                self.ds, axis, ix=self.p1[0], iy=self.p1[1]
            )
        )

    def export_image_file(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=RASTER_DATA_FILETYPES,
        )
        if not path:
            return
        try:
            save_display_image(
                path,
                self.ds.data,
                display_colormap(
                    self.cmap.get(), self.invert_colors.get()
                ),
            )
        except ValueError as error:
            messagebox.showerror("Export Image", str(error), parent=self)

    def serial_state(self):
        return {
            **self.state,
            "p1": self.p1, "p2": self.p2, "second": self.second,
            "cmap": self.cmap.get(), "invert_colors": self.invert_colors.get(),
            "reference": self.reference,
            "show_reference": self.show_reference.get(),
            "ignore_reference": self.ignore_reference.get(),
            "axis_ids": self.axis_ids,
            "inversions": [variable.get() for variable in self.invert_vars],
            "shown_instructions": sorted(self.shown_instructions),
        }

    def restore_state(self, state):
        self.p1 = list(state.pop("p1"))
        self.p2 = list(state.pop("p2"))
        self.second_var.set(state.pop("second"))
        self.cmap.set(state.pop("cmap"))
        self.invert_colors.set(state.pop("invert_colors"))
        self.reference = state.pop("reference")
        self.show_reference.set(state.pop("show_reference"))
        self.ignore_reference.set(state.pop("ignore_reference"))
        self.axis_ids = state.pop("axis_ids")
        inversions = state.pop("inversions")
        for variable, value in zip(self.invert_vars, inversions):
            variable.set(value)
        self.applied_inversions = list(inversions)
        self.shown_instructions = set(state.pop("shown_instructions"))
        super().restore_state(state)


class HyperspectralView(BaseView):
    """Coordinated hyperspectral image view; data storage order is (y, x, z)."""
    def __init__(self, app, ds):
        super().__init__(app, ds)
        nz = ds.data.shape[2]
        self.pos = initial_hyperspectral_position(ds.data.shape, ds.axes[2])
        self.drag = self.range_channel = None
        self.reference = None
        self.shown_instructions: set[str] = set()
        self.mode = tk.StringVar(value="Z Slice")
        self.cmap = tk.StringVar(value="viridis")
        self.invert_z = tk.BooleanVar(value=False)
        self.show_reference = tk.BooleanVar(value=True)
        self.show_profile_cursors = tk.BooleanVar(value=True)
        self.ignore_reference = tk.BooleanVar(value=False)
        self.channels = {
            c: {"mode": "Dark", "range": [max(0, nz // 3 - 2), min(nz - 1, nz // 3 + 2)]}
            for c in "RGB"
        }
        self.channel_vars = {c: tk.StringVar(value="Dark") for c in "RGB"}
        self.combine = tk.StringVar(value="Average")
        self.axis_ids = list(ds.labels)
        self.axis_vars = [tk.StringVar(value=v) for v in self.axis_ids]
        self.invert_vars = [tk.BooleanVar(value=False) for _ in range(3)]
        self.applied_inversions = [False, False, False]

        ttk.Checkbutton(
            self.view_controls, text="Show Profile Cursors",
            variable=self.show_profile_cursors, command=self.draw
        ).pack(side="left")
        ttk.Checkbutton(
            self.view_controls, text="Show Contrast Reference",
            variable=self.show_reference, command=self.draw
        ).pack(side="left", padx=4)

        # Fixed layout prevents tick-label width changes from shifting quadrants.
        self.fig.set_constrained_layout(False)
        self.fig.subplots_adjust(
            left=.08, right=.97, bottom=.08, top=.93, wspace=.24, hspace=.28
        )
        outer = self.fig.add_gridspec(2, 2)
        self.ax_xy = self.fig.add_subplot(outer[1, 0])
        left_top = outer[0, 0].subgridspec(2, 1, height_ratios=[1, 2])
        self.ax_x = self.fig.add_subplot(left_top[0], sharex=self.ax_xy)
        self.ax_xz = self.fig.add_subplot(left_top[1], sharex=self.ax_xy)
        self.ax_spec = self.fig.add_subplot(outer[0, 1])
        right_bottom = outer[1, 1].subgridspec(1, 2, width_ratios=[2, 1])
        self.ax_yz = self.fig.add_subplot(right_bottom[0], sharey=self.ax_xy)
        self.ax_y = self.fig.add_subplot(right_bottom[1], sharey=self.ax_xy)

        # Row 1: controls shared by both display modes.
        self.display_controls = ttk.Frame(self)
        self.display_controls.pack(fill="x", after=self.bottom)
        ttk.Label(self.display_controls, text="Display:").pack(side="left", padx=(4, 1))
        self.mode_box = ttk.Combobox(
            self.display_controls, textvariable=self.mode,
            values=["Z Slice", "RGB Channels"], width=13, state="readonly"
        )
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>", self.on_mode)
        ttk.Button(
            self.display_controls, text="Contrast Reference",
            command=self.reference_mode
        ).pack(side="left", padx=4)
        ttk.Checkbutton(
            self.display_controls, text="Ignore Contrast Reference",
            variable=self.ignore_reference, command=self.draw
        ).pack(side="left", padx=4)
        # Row 2: controls specific to the selected display mode.
        self.mode_controls = ttk.Frame(self)
        self.mode_controls.pack(fill="x", after=self.display_controls)
        self.z_controls = ttk.Frame(self.mode_controls)
        ttk.Label(self.z_controls, text="Colormap:").pack(side="left")
        self.cmap_button = grouped_colormap_button(
            self.z_controls, self.cmap
        )
        self.cmap_button.pack(side="left")
        ttk.Checkbutton(
            self.z_controls, text="Invert Colors",
            variable=self.invert_z, command=self.draw
        ).pack(side="left", padx=6)

        self.rgb_controls = ttk.Frame(self.mode_controls)
        ttk.Label(self.rgb_controls, text="Value:").pack(
            side="left", padx=(5, 1)
        )
        combine_box = ttk.Combobox(
            self.rgb_controls, textvariable=self.combine,
            values=["Average", "Maximum", "Minimum"],
            width=10, state="readonly"
        )
        combine_box.pack(side="left", padx=(0, 5))
        combine_box.bind("<<ComboboxSelected>>", lambda _event: self.draw())
        for c in "RGB":
            ttk.Label(self.rgb_controls, text=f"{c}:").pack(side="left", padx=(5, 1))
            box = ttk.Combobox(
                self.rgb_controls, textvariable=self.channel_vars[c],
                values=["Dark", "Bright", "Range", "Inverted Range"],
                width=13, state="readonly"
            )
            box.pack(side="left")
            box.bind("<<ComboboxSelected>>", lambda _e, channel=c: self.channel_changed(channel))
            ttk.Button(
                self.rgb_controls, text=f"Select {c} range",
                command=lambda channel=c: self.select_channel_range(channel)
            ).pack(side="left", padx=(1, 4))

        self.axis_controls = ttk.Frame(self)
        self.axis_controls.pack(fill="x", after=self.mode_controls)
        for i, role in enumerate(("X axis", "Y axis", "Z axis")):
            ttk.Label(self.axis_controls, text=role + ":").pack(side="left", padx=(5, 1))
            ttk.Combobox(
                self.axis_controls, textvariable=self.axis_vars[i],
                values=self.axis_ids, width=11, state="readonly"
            ).pack(side="left")
            ttk.Checkbutton(
                self.axis_controls, text="Invert", variable=self.invert_vars[i]
            ).pack(side="left")
        ttk.Button(self.axis_controls, text="Apply axes", command=self.apply_axes).pack(side="left", padx=4)
        for text, command in (
            ("Spectrum to New View", self.export_curve),
            ("X Profile to New View", lambda: self.export_profile("x")),
            ("Y Profile to New View", lambda: self.export_profile("y")),
            ("2D Slice to New View", self.export_slice),
        ):
            ttk.Button(
                self.new_view_bar, text=text, command=command
            ).pack(side="right")
        ttk.Button(
            self.export_bar, text="Export Image to File",
            command=self.export_image_file,
        ).pack(side="right", before=self.export_data_button)

        self.cmap.trace_add("write", lambda *_: self.draw())
        self.canvas.mpl_connect("button_press_event", self.press)
        self.canvas.mpl_connect("motion_notify_event", self.motion)
        self.canvas.mpl_connect("button_release_event", self.release)
        self.title_var.trace_add("write", lambda *_: self.draw())
        self.on_mode()

    def on_mode(self, _event=None):
        if self.mode.get() == "RGB Channels":
            self.z_controls.pack_forget()
            self.rgb_controls.pack(fill="x")
        else:
            self.rgb_controls.pack_forget()
            self.z_controls.pack(fill="x")
        self.draw()

    def channel_changed(self, channel):
        self.channels[channel]["mode"] = self.channel_vars[channel].get()
        self.draw()

    def select_channel_range(self, channel):
        self.range_channel = channel
        if self.channels[channel]["mode"] not in ("Range", "Inverted Range"):
            self.channels[channel]["mode"] = "Range"
            self.channel_vars[channel].set("Range")
        if consume_first_use(self.shown_instructions, "channel_range"):
            messagebox.showinfo(
                "Channel range",
                "Drag repeatedly across the Z curve to set the selected channel's range."
            )
        self.draw()

    def channel_image(self):
        result = []
        for channel in "RGB":
            setting = self.channels[channel]
            mode = setting["mode"]
            if mode == "Dark":
                values = np.zeros(self.ds.data.shape[:2])
            elif mode == "Bright":
                values = np.ones(self.ds.data.shape[:2])
            else:
                lo, hi = sorted(setting["range"])
                chunk = self.ds.data[:, :, lo:hi + 1]
                values = collapse_spectral_range(chunk, self.combine.get())
                values = normalize_display_channel(
                    values,
                    reference=self.reference,
                    use_reference=not self.ignore_reference.get(),
                )
                if mode == "Inverted Range":
                    values = 1 - values
            result.append(values)
        return np.clip(np.stack(result, axis=2), 0, 1)

    def draw(self):
        for ax in (self.ax_x, self.ax_xz, self.ax_spec, self.ax_xy, self.ax_yz, self.ax_y):
            ax.clear()
        data = self.ds.data
        x, y, z = self.ds.axes
        ix, iy, iz = self.pos
        sections = hyperspectral_sections(data, ix, iy, iz)
        xz_plane, yz_plane, z_limits = orient_spectral_planes(sections, z)
        image = sections["xy"] if self.mode.get() == "Z Slice" else self.channel_image()
        scalar_colormap = display_colormap(self.cmap.get(), self.invert_z.get())
        shown = self.ax_xy.imshow(
            image, origin="lower", aspect="auto",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap=scalar_colormap if image.ndim == 2 else None
        )
        limits = contrast_reference_limits(
            data, iz, None if self.ignore_reference.get() else self.reference
        )
        if image.ndim == 2 and limits:
            shown.set_clim(*limits)
        if self.reference and self.show_reference.get():
            from matplotlib.patches import Rectangle
            (x0, y0), (x1, y1) = self.reference
            xa, xb = sorted((x[x0], x[x1]))
            ya, yb = sorted((y[y0], y[y1]))
            self.ax_xy.add_patch(Rectangle((xa, ya), xb-xa, yb-ya, fill=False, color="yellow", lw=1.5))

        self.ax_spec.plot(z, sections["z_profile"], color="green")
        cursor_visibility = profile_cursor_visibility(
            self.show_profile_cursors.get()
        )
        if cursor_visibility["curve"]:
            self.ax_spec.axvline(z[iz], color="lime")
        self.ax_x.plot(x, sections["x_profile"], color="red")
        self.ax_xz.imshow(
            xz_plane, origin="lower", aspect="auto",
            extent=[x[0], x[-1], *z_limits], cmap=scalar_colormap
        )
        if cursor_visibility["xz"]:
            self.ax_xz.axhline(z[iz], color="cyan")
        self.ax_yz.imshow(
            yz_plane, origin="lower", aspect="auto",
            extent=[*z_limits, y[0], y[-1]], cmap=scalar_colormap
        )
        if cursor_visibility["yz"]:
            self.ax_yz.axvline(z[iz], color="cyan")
        self.ax_y.plot(sections["y_profile"], y, color="red")
        self.ax_x.tick_params(axis="x", labelbottom=False)
        self.ax_xz.tick_params(axis="x", labelbottom=False)
        self.ax_yz.tick_params(axis="y", labelleft=False)
        self.ax_y.tick_params(axis="y", labelleft=False)
        if self.state["cursors"]:
            self.ax_xy.axvline(x[ix], color="cyan")
            self.ax_xy.axhline(y[iy], color="cyan")
        if self.mode.get() == "RGB Channels":
            for channel in "RGB":
                setting = self.channels[channel]
                if setting["mode"] in ("Range", "Inverted Range"):
                    lo, hi = sorted(setting["range"])
                    self.ax_spec.axvspan(z[lo], z[hi], alpha=.13, color=channel.lower())
        self.fig.suptitle(self.ds.title)
        self.refresh_meta({
            "Cursor X": x[ix], "Cursor Y": y[iy], "Slice Z": z[iz],
            "RGB channels": self.channels,
            "Contrast reference": self.reference,
            "Contrast reference shown": self.show_reference.get(),
            "Contrast reference ignored": self.ignore_reference.get(),
        })
        self.canvas.draw_idle()

    def image_point(self, event):
        x, y, _ = self.ds.axes
        return [
            int(np.argmin(abs(x - event.xdata))),
            int(np.argmin(abs(y - event.ydata))),
        ]

    def press(self, event):
        x, y, z = self.ds.axes
        if event.inaxes == self.ax_xy and event.xdata is not None:
            point = self.image_point(event)
            if self.drag == "reference-ready":
                self.reference = [point, point.copy()]
                self.drag = "reference"
            else:
                self.pos[:2] = point
                self.drag = "xy"
        elif event.inaxes == self.ax_spec and event.xdata is not None:
            index = nearest_axis_index(z, event.xdata)
            if self.mode.get() == "RGB Channels" and self.range_channel:
                self.channels[self.range_channel]["range"][0] = index
                self.channels[self.range_channel]["range"][1] = index
                self.drag = "range"
            else:
                self.pos[2] = index
                self.drag = "z"
        elif (
            self.show_profile_cursors.get()
            and event.inaxes == self.ax_xz
            and event.ydata is not None
        ):
            self.pos[2] = nearest_axis_index(z, event.ydata)
            self.drag = "z-xz"
        elif (
            self.show_profile_cursors.get()
            and event.inaxes == self.ax_yz
            and event.xdata is not None
        ):
            self.pos[2] = nearest_axis_index(z, event.xdata)
            self.drag = "z-yz"
        self.draw()

    def motion(self, event):
        if event.xdata is None:
            return
        if self.drag == "reference" and event.inaxes == self.ax_xy:
            self.reference[1] = self.image_point(event)
            self.draw()
        elif self.drag == "xy" and event.inaxes == self.ax_xy:
            self.pos[:2] = self.image_point(event)
            self.draw()
        elif self.drag == "z" and event.inaxes == self.ax_spec:
            self.pos[2] = nearest_axis_index(self.ds.axes[2], event.xdata)
            self.draw()
        elif (
            self.drag == "z-xz"
            and event.inaxes == self.ax_xz
            and event.ydata is not None
        ):
            self.pos[2] = nearest_axis_index(self.ds.axes[2], event.ydata)
            self.draw()
        elif (
            self.drag == "z-yz"
            and event.inaxes == self.ax_yz
            and event.xdata is not None
        ):
            self.pos[2] = nearest_axis_index(self.ds.axes[2], event.xdata)
            self.draw()
        elif self.drag == "range" and event.inaxes == self.ax_spec and self.range_channel:
            self.channels[self.range_channel]["range"][1] = int(
                np.argmin(abs(self.ds.axes[2] - event.xdata))
            )
            self.draw()

    def release(self, _event):
        self.drag = None

    def reference_mode(self):
        self.drag = "reference-ready"
        if consume_first_use(self.shown_instructions, "contrast_reference"):
            messagebox.showinfo(
                "Contrast Reference",
                "Drag a rectangle on the XY Slice. Its minimum and maximum will remain "
                "the contrast limits as you move through Z. Select this again to replace it."
            )

    def apply_axes(self):
        requested = [var.get() for var in self.axis_vars]
        if len(set(requested)) != 3:
            messagebox.showerror("Axes", "X, Y, and Z must use three different source axes.")
            return
        current = list(self.axis_ids)
        source_indices = [current.index(name) for name in requested]
        axis_to_data_dimension = {0: 1, 1: 0, 2: 2}
        permutation = [
            axis_to_data_dimension[source_indices[1]],
            axis_to_data_dimension[source_indices[0]],
            axis_to_data_dimension[source_indices[2]],
        ]
        self.ds.data = np.transpose(self.ds.data, permutation)
        self.ds.axes = [self.ds.axes[i] for i in source_indices]
        self.ds.labels = [self.ds.labels[i] for i in source_indices]
        self.ds.units = [self.ds.units[i] for i in source_indices]
        self.axis_ids = requested
        for i in range(3):
            desired = self.invert_vars[i].get()
            old_index = source_indices[i]
            was_inverted = self.applied_inversions[old_index]
            if desired != was_inverted:
                self.ds.axes[i] = -self.ds.axes[i]
        self.applied_inversions = [v.get() for v in self.invert_vars]
        ny, nx, nz = self.ds.data.shape
        self.pos = initial_hyperspectral_position(
            self.ds.data.shape, self.ds.axes[2]
        )
        self.reference = None
        self.draw()

    def export_curve(self):
        ix, iy, _ = self.pos
        values = hyperspectral_sections(self.ds.data, ix, iy, self.pos[2])["z_profile"]
        derived = DataSet(
            "curve", values.copy(), [self.ds.axes[2].copy()],
            f"{self.ds.title} curve", self.ds.source, {},
            [self.ds.units[2], "a.u."], [self.ds.labels[2], "Signal"],
            parent=self.ds.uid, derivation={"x": ix, "y": iy}
        )
        self.app.open_dataset(derived)

    def export_slice(self):
        self.app.open_dataset(
            hyperspectral_slice_dataset(self.ds, self.pos[2])
        )

    def current_display_image(self) -> np.ndarray:
        if self.mode.get() == "RGB Channels":
            return self.channel_image().copy()
        return self.ds.data[:, :, self.pos[2]].copy()

    def export_image_file(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=RASTER_DATA_FILETYPES,
        )
        if not path:
            return
        colormap = (
            display_colormap(self.cmap.get(), self.invert_z.get())
            if self.mode.get() == "Z Slice" else None
        )
        try:
            save_display_image(
                path, self.current_display_image(), colormap
            )
        except ValueError as error:
            messagebox.showerror(
                "Export Image", str(error), parent=self
            )

    def export_profile(self, axis):
        ix, iy, iz = self.pos
        sections = hyperspectral_sections(self.ds.data, ix, iy, iz)
        values = sections[f"{axis}_profile"]
        coordinate = self.ds.axes["xy".index(axis)]
        derived = DataSet(
            "curve", values.copy(), [coordinate.copy()],
            f"{self.ds.title} {axis.upper()} profile", self.ds.source, {},
            [self.ds.units["xy".index(axis)], "a.u."], [axis.upper(), "Signal"],
            parent=self.ds.uid, derivation={"z": iz, "x": ix, "y": iy}
        )
        self.app.open_dataset(derived)

    def serial_state(self):
        return {
            **self.state, "pos": self.pos, "mode": self.mode.get(),
            "cmap": self.cmap.get(), "reference": self.reference,
            "channels": self.channels, "combine": self.combine.get(),
            "axis_ids": self.axis_ids,
            "inversions": [v.get() for v in self.invert_vars],
            "invert_z": self.invert_z.get(),
            "show_reference": self.show_reference.get(),
            "ignore_reference": self.ignore_reference.get(),
            "show_profile_cursors": self.show_profile_cursors.get(),
            "shown_instructions": sorted(self.shown_instructions),
        }

    def restore_state(self, state):
        self.pos = list(state.pop("pos"))
        self.mode.set(state.pop("mode"))
        self.cmap.set(state.pop("cmap"))
        self.reference = state.pop("reference")
        self.channels = state.pop("channels")
        for channel in "RGB":
            self.channel_vars[channel].set(self.channels[channel]["mode"])
        self.combine.set(state.pop("combine"))
        self.invert_z.set(state.pop("invert_z"))
        self.show_reference.set(state.pop("show_reference"))
        self.ignore_reference.set(state.pop("ignore_reference"))
        self.show_profile_cursors.set(state.pop("show_profile_cursors"))
        self.shown_instructions = set(state.pop("shown_instructions"))
        self.axis_ids = state.pop("axis_ids")
        inversions = state.pop("inversions")
        for var, value in zip(self.invert_vars, inversions):
            var.set(value)
        self.applied_inversions = list(inversions)
        self.on_mode()
        super().restore_state(state)


class ImportDialog(tk.Toplevel):
    PARSERS = {
        "curve": "two_column",
        "image": "matrix",
        "hyperspectral": "coordinate_hyperspectral",
    }

    def __init__(self, app, path, detections):
        super().__init__(app)
        self.app = app
        self.path = path
        self.detections = detections
        self.title("Import interpretation")
        self.transient(app)
        self.grab_set()

        detection = detections[0]
        summary = (
            f"{Path(path).name}\n"
            f"Detected: {detection.kind}, shape {detection.shape}, "
            f"confidence {detection.confidence:.0%}\n"
            f"{detection.details}"
        )
        ttk.Label(self, text=summary, justify="left").pack(
            padx=14, pady=10
        )
        self.kind = tk.StringVar(value=detection.kind)
        self.row = tk.IntVar(value=0)
        ttk.Label(self, text="Import as:").pack(anchor="w", padx=14)
        self.kind_box = ttk.Combobox(
            self, textvariable=self.kind, values=list(self.PARSERS),
            state="readonly",
        )
        self.kind_box.pack(fill="x", padx=14)
        self.kind_box.bind(
            "<<ComboboxSelected>>", self.update_row_control
        )
        self.row_label = ttk.Label(
            self, text="Curve row index (for row-based files):"
        )
        self.row_label.pack(anchor="w", padx=14, pady=(8, 0))
        self.row_box = ttk.Spinbox(
            self, from_=0, to=100000, textvariable=self.row
        )
        self.row_box.pack(fill="x", padx=14)
        ttk.Button(
            self, text="Import", command=self.import_data
        ).pack(pady=12)
        self.update_row_control()

    def update_row_control(self, _event=None):
        row_based = (
            self.kind.get() == "curve"
            and any(
                detection.kind == "curve"
                and detection.parser == "row_line"
                for detection in self.detections
            )
        )
        state = "normal" if row_based else "disabled"
        self.row_box.configure(state=state)

    def import_data(self):
        kind = self.kind.get()
        candidates = [
            detection for detection in self.detections
            if detection.kind == kind
        ]
        detection = (
            candidates[0] if candidates else
            Detection(
                kind, .2, (), "Manual override", self.PARSERS[kind]
            )
        )
        try:
            dataset = load_text(self.path, detection, self.row.get())
            self.app.open_dataset(dataset)
            self.destroy()
        except Exception as error:
            messagebox.showerror(
                "Import failed", str(error), parent=self
            )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HyperScope")
        self.geometry("1350x850")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.datasets = {}
        self.views = {}
        self.dirty = False

        left = ttk.Frame(self, width=240)
        left.pack(side="left", fill="y")
        main = ttk.Frame(self)
        main.pack(side="left", fill="both", expand=True)
        ttk.Button(
            left, text="Click to import files", command=self.pick
        ).pack(fill="x", padx=8, pady=8, ipady=12)
        ttk.Label(left, text="Open Views").pack(anchor="w", padx=8)
        self.tree = ttk.Treeview(left, show="tree")
        self.tree.pack(fill="both", expand=True, padx=8)
        self.tree.bind("<<TreeviewSelect>>", self.select_tree)
        ttk.Button(
            left, text="Save Session", command=self.save_session
        ).pack(fill="x", padx=8, pady=4)
        ttk.Button(
            left, text="Open Session", command=self.open_session
        ).pack(fill="x", padx=8, pady=(0, 8))
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True)
    def pick(self):
        paths=filedialog.askopenfilenames(
            filetypes=[
                ("Text data or session", "*.txt *.csv *.hscope"),
                ("Text data", "*.txt *.csv"),
                ("HyperScope session", "*.hscope"),
                ("All files", "*.*"),
            ]
        )
        self.import_paths(paths)
    def import_paths(self, paths):
        for path in paths:
            if str(path).lower().endswith(".hscope"):
                self.open_session(path)
                continue
            try:
                ImportDialog(
                    self, str(path), detect_text(str(path))
                )
            except Exception as error:
                messagebox.showerror(
                    "Detection failed", f"{path}\n{error}"
                )
    def open_dataset(self, dataset):
        view_classes = {
            "curve": CurveView,
            "image": ImageView,
            "hyperspectral": HyperspectralView,
        }
        self.datasets[dataset.uid] = dataset
        view = view_classes[dataset.kind](self, dataset)
        self.views[dataset.uid] = view
        self.notebook.add(view, text=dataset.title)
        self.notebook.select(view)
        self.dirty = True
        self.refresh_tree()
    def delete_view(self, uid):
        dataset = self.datasets.get(uid)
        view = self.views.pop(uid, None)
        if view is not None:
            self.notebook.forget(view)
            view.destroy()
        if dataset is not None:
            for child in self.datasets.values():
                if child.parent == uid:
                    child.parent = dataset.parent
            self.datasets.pop(uid, None)
        self.dirty = True
        self.refresh_tree()
    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())

        def add(parent=None):
            for dataset in self.datasets.values():
                if dataset.parent == parent:
                    self.tree.insert(
                        parent or "", "end", iid=dataset.uid,
                        text=dataset.title, open=True,
                    )
                    add(dataset.uid)

        add()
        for view in self.views.values():
            try:
                self.notebook.tab(view, text=view.ds.title)
            except tk.TclError:
                pass
    def select_tree(self, _event):
        selection = self.tree.selection()
        uid = selection[0] if selection else ""
        if uid in self.views:
            self.notebook.select(self.views[uid])
    def save_session(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".hscope",
            filetypes=[("HyperScope session", "*.hscope")],
        )
        if not path:
            return False
        payload = {
            "version": APP_VERSION,
            "datasets": self.datasets,
            "states": {
                uid: view.serial_state()
                for uid, view in self.views.items()
            },
            "selected": (
                self.notebook.index("current") if self.views else 0
            ),
        }
        with open(path, "wb") as session_file:
            pickle.dump(payload, session_file, protocol=5)
        self.dirty = False
        return True
    def open_session(self, path=None):
        path = path or filedialog.askopenfilename(
            filetypes=[("HyperScope session", "*.hscope")]
        )
        if not path:
            return
        with open(path, "rb") as session_file:
            payload = pickle.load(session_file)
        try:
            validate_session_payload(payload)
        except ValueError as error:
            messagebox.showerror("Cannot open session", str(error))
            return
        for view in self.views.values():
            self.notebook.forget(view)
        self.datasets = {}
        self.views = {}
        pending = dict(payload["datasets"])
        while pending:
            progressed = False
            for uid, dataset in list(pending.items()):
                if not dataset.parent or dataset.parent in self.datasets:
                    self.open_dataset(dataset)
                    pending.pop(uid)
                    progressed = True
            if not progressed:
                for dataset in pending.values():
                    dataset.parent = None
                    self.open_dataset(dataset)
                break
        for uid, state in payload["states"].items():
            if uid in self.views:
                self.views[uid].restore_state(dict(state))
        if self.views:
            selected = min(
                payload["selected"], len(self.views) - 1
            )
            self.notebook.select(selected)
        self.dirty = False
    def close(self):
        if not self.dirty:
            self.destroy()
            return
        dialog = tk.Toplevel(self)
        dialog.title("Close session?")
        dialog.transient(self)
        dialog.grab_set()
        ttk.Label(
            dialog, text="Save the exact session state before closing?"
        ).pack(padx=24, pady=16)
        buttons = ttk.Frame(dialog)
        buttons.pack(pady=(0, 16))
        ttk.Button(
            buttons, text="Save Session",
            command=lambda: (
                self.destroy() if self.save_session() else None
            ),
        ).pack(side="left", padx=4)
        ttk.Button(
            buttons, text="End Without Saving", command=self.destroy
        ).pack(side="left", padx=4)
        ttk.Button(
            buttons, text="Cancel", command=dialog.destroy
        ).pack(side="left", padx=4)


if __name__ == "__main__":
    App().mainloop()
