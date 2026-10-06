"""Tkinter front end for run_pipeline.py and align_outputs.py, one tab each.

Each tab's form is generated from its script's build_parser(), so any flag
added to either CLI shows up here automatically with its default and help
text (hover a field to see it). Layout follows the parser's argument groups:
every checkbox is collected into one Options box, and each group becomes its
own box with input fields and output fields (dest starting "output_") side by
side. Scripts run as subprocesses rather than in-process, so a native crash in
OpenSim/ezc3d ends that run without taking the GUI down.

Launch from the Theia2OpenSim conda environment:
    python src/Python/run_pipeline_gui.py
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

import align_outputs
import run_pipeline
from workflow_utils import DEFAULT_GCS_ROTATIONS

PIPELINE_SCRIPT = Path(run_pipeline.__file__).resolve()
ALIGN_SCRIPT = Path(align_outputs.__file__).resolve()
INPUT_BG = "#ffffff"
OUTPUT_BG = "#e6f2e6"
OPTIONS_PER_ROW = 5
GCS_ROTATION_SLOTS = 3
AXIS_CHOICES = ("none", "X", "Y", "Z")
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PLATE_COUNT_PROBE = (
    "import sys, ezc3d\n"
    "params = ezc3d.c3d(sys.argv[1])['parameters']\n"
    "print(int(params['FORCE_PLATFORM']['USED']['value'][0]) if 'FORCE_PLATFORM' in params else 0)\n"
)


def _field_kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):
        return "flag"
    if action.metavar == "AXIS:DEG":
        return "rotation"
    if action.metavar == "N:L|R":
        return "plate_foot"
    if action.nargs == "*":
        return "list"
    if action.type is Path:
        return "path"
    return "value"


def _is_output(action: argparse.Action) -> bool:
    return action.dest.startswith("output_")


def _default_text(action: argparse.Action) -> str:
    if action.default is None:
        return ""
    if isinstance(action.default, (list, tuple)):
        return " ".join(str(v) for v in action.default)
    return str(action.default)


class _Tooltip:
    def __init__(self, widget: tk.Widget, text: str | None):
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event) -> None:
        if self.tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", wraplength=480, background="#ffffe0",
                 relief="solid", borderwidth=1, padx=6, pady=4).pack()

    def _hide(self, _event) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class _RotationField:
    """Ordered axis/angle slots for --gcs-rot.

    Rotations compose, so their order matters (Z then X is not X then Z) --
    each slot picks its own axis rather than having fixed X/Y/Z boxes. Slots
    set to "none" are skipped; all "none" means no rotation at all.
    """

    def __init__(self, parent: tk.Widget, help_text: str | None):
        self.frame = ttk.Frame(parent)
        self.slots: list[tuple[tk.StringVar, tk.StringVar]] = []
        for index in range(GCS_ROTATION_SLOTS):
            if index:
                ttk.Label(self.frame, text="then").pack(side="left", padx=10)
            axis = tk.StringVar()
            angle = tk.StringVar()
            combo = ttk.Combobox(self.frame, textvariable=axis, values=AXIS_CHOICES, state="readonly", width=5)
            combo.pack(side="left")
            entry = tk.Entry(self.frame, textvariable=angle, width=7, relief="solid", borderwidth=1)
            entry.pack(side="left", padx=(4, 0), ipady=2)
            ttk.Label(self.frame, text="°").pack(side="left", padx=(2, 0))
            axis.trace_add("write", lambda *_args, axis=axis, entry=entry:
                           entry.configure(state="disabled" if axis.get() == "none" else "normal"))
            _Tooltip(combo, help_text)
            _Tooltip(entry, help_text)
            self.slots.append((axis, angle))
        self.reset()

    def reset(self) -> None:
        for index, (axis, angle) in enumerate(self.slots):
            if index < len(DEFAULT_GCS_ROTATIONS):
                default_axis, default_deg = DEFAULT_GCS_ROTATIONS[index]
                axis.set(default_axis)
                angle.set(f"{default_deg:g}")
            else:
                axis.set("none")
                angle.set("")

    def copy_from(self, other: _RotationField) -> None:
        for (axis, angle), (other_axis, other_angle) in zip(self.slots, other.slots):
            axis.set(other_axis.get())
            angle.set(other_angle.get())

    def tokens(self) -> list[str]:
        tokens = []
        for index, (axis, angle) in enumerate(self.slots, start=1):
            if axis.get() == "none":
                continue
            text = angle.get().strip()
            try:
                float(text)
            except ValueError:
                raise ValueError(f"--gcs-rot rotation {index} ({axis.get()}): angle must be a number, "
                                 f"got {text!r}.") from None
            tokens.append(f"{axis.get()}:{text}")
        return tokens


class _PlateFootField:
    """Left/Right checkboxes per force plate, for --plate-foot.

    One row per plate in the GRF C3D, re-read whenever that path changes.
    The plate count is read in a subprocess (like the scripts themselves), so
    a malformed C3D can't crash the GUI. Ticking one box unticks the other;
    leaving both unticked keeps align_outputs' auto-detection for that plate.
    """

    def __init__(self, parent: tk.Widget, help_text: str | None, on_rows_changed: Callable[[], None]):
        self.frame = ttk.Frame(parent)
        self.help_text = help_text
        self.on_rows_changed = on_rows_changed
        self.status = ttk.Label(self.frame, foreground="#555555", wraplength=420)
        self.status.pack(anchor="w")
        self.grid = ttk.Frame(self.frame)
        self.grid.pack(anchor="w")
        self.rows: dict[int, tuple[tk.BooleanVar, tk.BooleanVar, list[tk.Widget]]] = {}
        self.path_var: tk.StringVar | None = None
        self.pending: str | None = None
        self.probe_id = 0
        self.results: queue.Queue[tuple[int, Path, int | None]] = queue.Queue()

    def watch(self, path_var: tk.StringVar) -> None:
        self.path_var = path_var
        path_var.trace_add("write", lambda *_args: self._schedule())
        self._schedule()

    def _schedule(self) -> None:
        # Debounced, so typing a path doesn't launch a probe per keystroke.
        if self.pending is not None:
            self.frame.after_cancel(self.pending)
        self.pending = self.frame.after(400, self._probe)

    def _probe(self) -> None:
        self.pending = None
        self.probe_id += 1
        path = Path(self.path_var.get().strip())
        if not path.is_file():
            self._set_plates(0, "--grf-c3d not found -- set it to list its force plates here.")
            return
        self.status.configure(text=f"Reading force plates from {path.name}...")
        threading.Thread(target=self._count_plates, args=(path, self.probe_id), daemon=True).start()
        self.frame.after(100, self._poll)

    def _count_plates(self, path: Path, probe_id: int) -> None:
        try:
            done = subprocess.run([sys.executable, "-c", _PLATE_COUNT_PROBE, str(path)], capture_output=True,
                                  text=True, timeout=60, creationflags=CREATE_NO_WINDOW)
            count = int(done.stdout.strip()) if done.returncode == 0 else None
        except (subprocess.TimeoutExpired, ValueError):
            count = None
        self.results.put((probe_id, path, count))

    def _poll(self) -> None:
        # Each probe starts one poll loop and puts one result, so each loop
        # consumes exactly one; results from a superseded path are dropped.
        try:
            probe_id, path, count = self.results.get_nowait()
        except queue.Empty:
            self.frame.after(100, self._poll)
            return
        if probe_id != self.probe_id:
            return
        if count is None:
            self._set_plates(0, f"Couldn't read force plates from {path.name} -- is it a valid C3D?")
        elif count == 0:
            self._set_plates(0, f"No force plates in {path.name}.")
        else:
            self._set_plates(count, f"{count} plate(s) in {path.name}. Unticked = auto-detect.")

    def _set_plates(self, count: int, message: str) -> None:
        self.status.configure(text=message)
        for plate in [p for p in self.rows if p > count]:
            for widget in self.rows.pop(plate)[2]:
                widget.destroy()
        for plate in range(1, count + 1):
            if plate in self.rows:
                continue
            left, right = tk.BooleanVar(), tk.BooleanVar()
            label = ttk.Label(self.grid, text=f"Plate {plate}:")
            left_box = ttk.Checkbutton(self.grid, text="Left", variable=left,
                                       command=lambda on=left, other=right: on.get() and other.set(False))
            right_box = ttk.Checkbutton(self.grid, text="Right", variable=right,
                                        command=lambda on=right, other=left: on.get() and other.set(False))
            label.grid(row=plate, column=0, sticky="w", padx=(0, 10), pady=1)
            left_box.grid(row=plate, column=1, sticky="w", padx=(0, 16))
            right_box.grid(row=plate, column=2, sticky="w")
            for widget in (label, left_box, right_box):
                _Tooltip(widget, self.help_text)
            self.rows[plate] = (left, right, [label, left_box, right_box])
        self.on_rows_changed()

    def tokens(self) -> list[str]:
        return [f"{plate}:{'L' if left.get() else 'R'}"
                for plate, (left, right, _widgets) in sorted(self.rows.items()) if left.get() or right.get()]

    def reset(self) -> None:
        for left, right, _widgets in self.rows.values():
            left.set(False)
            right.set(False)


class ScriptTab:
    """One script's form, run controls, and live log, in a notebook tab."""

    def __init__(self, parent: tk.Widget, build_parser: Callable[[], argparse.ArgumentParser],
                 script: Path, description: str):
        self.frame = ttk.Frame(parent, padding=(0, 6, 0, 0))
        self.build_parser = build_parser
        self.script = script
        # (action, kind, variable, entry widget or None). For kind "rotation"
        # the variable slot holds a _RotationField instead of a tk.Variable.
        self.fields: list[tuple[argparse.Action, str, tk.Variable | _RotationField, tk.Entry | None]] = []
        self.process: subprocess.Popen | None = None
        self.stopped = False
        self.output_queue: queue.Queue[str | int] = queue.Queue()
        self.laid_out = False

        self.paned = ttk.PanedWindow(self.frame, orient="vertical")
        self.paned.pack(fill="both", expand=True)
        self.paned.add(self._build_form(self.paned, description), weight=3)
        self.paned.add(self._build_run_panel(self.paned), weight=2)

        # Plate rows follow whichever GRF C3D this tab's --grf-c3d points at.
        for _action, kind, var, _entry in self.fields:
            if kind == "plate_foot":
                var.watch(self.var("grf_c3d"))

        # Sash position and Entry scrolling both need the real layout, which
        # a notebook tab only gets once it's first shown.
        self.paned.bind("<Map>", lambda _e: self.frame.after(50, self._after_layout), add="+")

    def var(self, dest: str) -> tk.Variable | _RotationField | _PlateFootField:
        return next(var for action, _kind, var, _entry in self.fields if action.dest == dest)

    def _after_layout(self) -> None:
        if self.laid_out:
            return
        if self.paned.winfo_height() <= 1:
            self.frame.after(50, self._after_layout)
            return
        self.laid_out = True
        self._fit_sash()
        self.show_path_ends()

    def _fit_sash(self) -> None:
        # Give the form the height it needs to show without scrolling, while
        # keeping at least ~150 px for the log.
        if not self.laid_out:
            return
        self.frame.update_idletasks()
        wanted = self.form_header.winfo_reqheight() + self.form_inner.winfo_reqheight() + 12
        self.paned.sashpos(0, max(200, min(wanted, self.paned.winfo_height() - 150)))

    def _build_form(self, parent: tk.Widget, description: str) -> ttk.Frame:
        outer = ttk.Frame(parent)
        self.form_header = ttk.Frame(outer)
        self.form_header.pack(fill="x", pady=(0, 4))
        # Empty unless the app adds tab-level actions (e.g. "Fill from step 1").
        self.top_bar = ttk.Frame(self.form_header)
        self.top_bar.pack(anchor="w")
        ttk.Label(self.form_header, text=description + "\nFields marked * are required. Output fields "
                  "(files the script writes) are shaded green. Hover any field for its description."
                  ).pack(anchor="w")

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        form = self.form_inner = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Scroll the form only while the pointer is over it, so the wheel
        # still scrolls the log pane normally.
        def _on_wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        sections = []
        for group in self.build_parser()._action_groups:
            actions = [a for a in group._group_actions if a.option_strings and a.dest != "help"]
            if actions:
                sections.append((group.title, actions))

        boxes = []
        for title, actions in sections:
            fields = [a for a in actions if _field_kind(a) != "flag"]
            if fields:
                boxes.append(self._build_section(form, title, fields))
        checkboxes = [a for _, actions in sections for a in actions if _field_kind(a) == "flag"]
        if checkboxes:
            boxes.insert(1, self._build_options(form, checkboxes))

        form.columnconfigure(0, weight=1)
        for index, box in enumerate(boxes):
            if index:
                ttk.Separator(form, orient="horizontal").grid(row=2 * index - 1, column=0, sticky="ew", pady=8)
            box.grid(row=2 * index, column=0, sticky="ew", padx=(0, 4))
        return outer

    def _build_options(self, parent: tk.Widget, actions: list[argparse.Action]) -> ttk.LabelFrame:
        box = ttk.LabelFrame(parent, text="Options", padding=(8, 4))
        for index, action in enumerate(actions):
            var = tk.BooleanVar(value=bool(action.default))
            check = ttk.Checkbutton(box, text=action.option_strings[-1], variable=var)
            check.grid(row=index // OPTIONS_PER_ROW, column=index % OPTIONS_PER_ROW, sticky="w", padx=(0, 28), pady=2)
            _Tooltip(check, action.help)
            self.fields.append((action, "flag", var, None))
        return box

    def _build_section(self, parent: tk.Widget, title: str, actions: list[argparse.Action]) -> ttk.LabelFrame:
        box = ttk.LabelFrame(parent, text=title, padding=(8, 4))
        columns = [("Inputs", [a for a in actions if not _is_output(a)]),
                   ("Outputs", [a for a in actions if _is_output(a)])]
        columns = [(name, column_actions) for name, column_actions in columns if column_actions]
        for col, (name, column_actions) in enumerate(columns):
            box.columnconfigure(col, weight=1, uniform="io")
            column = ttk.LabelFrame(box, text=name, padding=(6, 2))
            column.grid(row=0, column=col, sticky="nsew", padx=(6 if col else 0, 0))
            column.columnconfigure(1, weight=1)
            for row, action in enumerate(column_actions):
                self._add_field(column, row, action)
        return box

    def _add_field(self, parent: tk.Widget, row: int, action: argparse.Action) -> None:
        kind = _field_kind(action)
        output = _is_output(action)
        label = ttk.Label(parent, text=action.option_strings[-1] + (" *" if action.required else ""))
        label.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        _Tooltip(label, action.help)

        if kind == "rotation":
            rotation = _RotationField(parent, action.help)
            rotation.frame.grid(row=row, column=1, columnspan=2, sticky="w", pady=2)
            self.fields.append((action, kind, rotation, None))
            return
        if kind == "plate_foot":
            plates = _PlateFootField(parent, action.help, on_rows_changed=self._fit_sash)
            plates.frame.grid(row=row, column=1, columnspan=2, sticky="w", pady=2)
            self.fields.append((action, kind, plates, None))
            return

        var = tk.StringVar(value=_default_text(action))
        entry = tk.Entry(parent, textvariable=var, relief="solid", borderwidth=1,
                         background=OUTPUT_BG if output else INPUT_BG)
        entry.grid(row=row, column=1, sticky="ew", pady=2, ipady=2)
        if kind == "path":
            ttk.Button(parent, text="Save as..." if output else "Browse...",
                       command=lambda: self._browse(action, var, entry)).grid(row=row, column=2, padx=(4, 0))
        _Tooltip(entry, action.help)
        self.fields.append((action, kind, var, entry))

    def _build_run_panel(self, parent: tk.Widget) -> ttk.Frame:
        panel = ttk.Frame(parent)
        self.buttons = ttk.Frame(panel)
        self.buttons.pack(fill="x", pady=(0, 4))
        self.run_button = ttk.Button(self.buttons, text=f"Run {self.script.name}", command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(self.buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        ttk.Button(self.buttons, text="Reset to defaults", command=self.reset).pack(side="left")
        ttk.Button(self.buttons, text="Clear log",
                   command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=4)
        self.status = ttk.Label(self.buttons, text="Ready")
        self.status.pack(side="right")

        self.log = ScrolledText(panel, height=14, font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        return panel

    def _browse(self, action: argparse.Action, var: tk.StringVar, entry: tk.Entry) -> None:
        options = {"title": action.option_strings[-1]}
        if var.get().strip():
            current = Path(var.get().strip())
            if current.parent.exists():
                options["initialdir"] = str(current.parent)
            options["initialfile"] = current.name
        if _is_output(action):
            path = filedialog.asksaveasfilename(**options)
        else:
            path = filedialog.askopenfilename(**options)
        if path:
            var.set(path)
            entry.xview_moveto(1.0)

    def show_path_ends(self) -> None:
        # Long paths are most informative at their end (the filename), but an
        # Entry shows the start.
        for _action, kind, _var, entry in self.fields:
            if kind == "path":
                entry.xview_moveto(1.0)

    def build_command(self) -> list[str]:
        """Translate the form into a command line for this tab's script.

        Empty optional fields are omitted, so the script falls back to its own
        default for them -- same as leaving the flag off on the CLI.
        """
        args: list[str] = []
        for action, kind, var, _entry in self.fields:
            flag = action.option_strings[-1]
            if kind == "flag":
                if var.get():
                    args.append(flag)
                continue
            if kind == "rotation":
                # Always passed: with every slot "none" this is a bare
                # --gcs-rot, which both scripts treat as no rotation.
                args.append(flag)
                args.extend(var.tokens())
                continue
            if kind == "plate_foot":
                # Omitted entirely when nothing's ticked: auto-detect every plate.
                tokens = var.tokens()
                if tokens:
                    args.append(flag)
                    args.extend(tokens)
                continue

            text = var.get().strip()
            if not text:
                if action.required:
                    raise ValueError(f"{flag} is required.")
                continue
            if kind == "list":
                args.append(flag)
                args.extend(text.split())
                continue
            if action.type in (int, float):
                try:
                    action.type(text)
                except ValueError:
                    kind_name = "an integer" if action.type is int else "a number"
                    raise ValueError(f"{flag} must be {kind_name}, got {text!r}.") from None
            args.extend([flag, text])
        return [sys.executable, "-u", str(self.script), *args]

    def run(self) -> None:
        try:
            command = self.build_command()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self._append("$ " + subprocess.list2cmdline(command) + "\n\n")
        self.stopped = False
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            creationflags=CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()

        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.configure(text="Running...")
        self.frame.after(100, self._poll_output)

    def _read_output(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            self.output_queue.put(line)
        self.output_queue.put(process.wait())

    def _poll_output(self) -> None:
        while True:
            try:
                item = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, int):
                self._finished(item)
                return
            self._append(item)
        self.frame.after(100, self._poll_output)

    def _finished(self, returncode: int) -> None:
        self.process = None
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.stopped:
            message = "Stopped"
        elif returncode == 0:
            message = "Finished successfully"
        else:
            message = f"Failed (exit code {returncode})"
        self.status.configure(text=message)
        self._append(f"\n[{message}]\n\n")

    def _append(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def stop(self) -> None:
        if self.process is not None:
            self.stopped = True
            self.process.terminate()

    def reset(self) -> None:
        for action, kind, var, _entry in self.fields:
            if kind in ("rotation", "plate_foot"):
                var.reset()
            else:
                var.set(bool(action.default) if kind == "flag" else _default_text(action))
        self.show_path_ends()


class PipelineGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Theia2OpenSim")
        root.geometry(f"1200x{min(950, root.winfo_screenheight() - 80)}")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.pipeline = ScriptTab(
            self.notebook, run_pipeline.build_parser, PIPELINE_SCRIPT,
            "Step 1: convert Theia3D C3D files into OpenSim inputs -- scaled model, kinematics .mot, "
            "and (optionally) an unaligned GRF .mot.")
        self.align = ScriptTab(
            self.notebook, align_outputs.build_parser, ALIGN_SCRIPT,
            "Step 2 (after step 1): align the kinematics and force C3Ds to their shared time window, then "
            "write the ExternalLoads and Inverse Dynamics setup XMLs.\n"
            "\"Fill from step 1\" copies the C3D files, rotation, force threshold, and scaled model "
            "from the first tab, with output names based on those C3Ds.")
        self.notebook.add(self.pipeline.frame, text="  1. Pipeline (run_pipeline.py)  ")
        self.notebook.add(self.align.frame, text="  2. Align IK + GRF, ID setup (align_outputs.py)  ")
        # A plain tk.Button, since the native Windows ttk theme ignores
        # background colors on ttk buttons.
        tk.Button(self.align.top_bar, text="Fill from step 1", command=self.fill_align_from_pipeline,
                  background="#2f6fd6", foreground="white", activebackground="#2558aa",
                  activeforeground="white", relief="flat", padx=12, pady=3, cursor="hand2",
                  font=("Segoe UI", 9, "bold")).pack(side="left", pady=(0, 6))

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def fill_align_from_pipeline(self) -> None:
        """Copy step 1's shared settings into step 2, so the two scripts can't
        silently disagree on source files, rotation, or the scaled model.

        Output names follow align_outputs.py's own default naming pattern,
        based on the C3D stems, in the directory step 1 writes its .mot to --
        so a new trial doesn't overwrite another trial's aligned files.
        """
        src, dst = self.pipeline, self.align
        ik_c3d = src.var("c3d").get().strip()
        grf_c3d = src.var("grf_c3d").get().strip()
        if not ik_c3d or not grf_c3d:
            messagebox.showerror("Fill from step 1", "Step 1 needs both --c3d and --grf-c3d filled in.")
            return

        dst.var("gcs_rot").copy_from(src.var("gcs_rot"))
        for dst_dest, src_dest in (("ik_c3d", "c3d"), ("grf_c3d", "grf_c3d"),
                                   ("grf_force_threshold", "grf_force_threshold"),
                                   ("scaled_model", "output_osim")):
            dst.var(dst_dest).set(src.var(src_dest).get())

        out_dir = Path(src.var("output_mot").get().strip()).parent
        ik_stem, grf_stem = Path(ik_c3d).stem, Path(grf_c3d).stem
        for dest, name in (("output_ik_mot", f"{ik_stem}_aligned.mot"),
                           ("output_grf_mot", f"{grf_stem}_grf_aligned.mot"),
                           ("output_external_loads", f"{grf_stem}_external_loads.xml"),
                           ("output_id_setup", f"{grf_stem}_inverse_dynamics_setup.xml")):
            dst.var(dest).set(str(out_dir / name))
        dst.show_path_ends()
        dst.status.configure(text="Filled from step 1")

    def on_close(self) -> None:
        running = [tab for tab in (self.pipeline, self.align) if tab.process is not None]
        if running:
            if not messagebox.askokcancel("Script running", "A run is still in progress. Stop it and quit?"):
                return
            for tab in running:
                tab.process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PipelineGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
