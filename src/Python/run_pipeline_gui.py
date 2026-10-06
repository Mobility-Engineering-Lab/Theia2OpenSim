"""Tkinter front end for run_pipeline.py.

The form is generated from run_pipeline.build_parser(), so any flag added to
the CLI shows up here automatically with its default and help text (hover a
field to see it). Layout follows the parser's argument groups: every
checkbox is collected into one Options box, and each group becomes its own
box with input fields and output fields (dest starting "output_") side by
side. The pipeline runs as a subprocess rather than in-process, so a native
crash in OpenSim/ezc3d ends that run without taking the GUI down.

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

from run_pipeline import build_parser
from workflow_utils import DEFAULT_GCS_ROTATIONS

PIPELINE_SCRIPT = Path(__file__).resolve().with_name("run_pipeline.py")
INPUT_BG = "#ffffff"
OUTPUT_BG = "#e6f2e6"
OPTIONS_PER_ROW = 5
GCS_ROTATION_SLOTS = 3
AXIS_CHOICES = ("none", "X", "Y", "Z")


def _field_kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):
        return "flag"
    if action.metavar == "AXIS:DEG":
        return "rotation"
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


class PipelineGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        # (action, kind, variable, entry widget or None). For kind "rotation"
        # the variable slot holds a _RotationField instead of a tk.Variable.
        self.fields: list[tuple[argparse.Action, str, tk.Variable | _RotationField, tk.Entry | None]] = []
        self.process: subprocess.Popen | None = None
        self.stopped = False
        self.output_queue: queue.Queue[str | int] = queue.Queue()

        root.title("Theia2OpenSim Pipeline")
        root.geometry(f"1200x{min(950, root.winfo_screenheight() - 80)}")

        self.paned = ttk.PanedWindow(root, orient="vertical")
        self.paned.pack(fill="both", expand=True, padx=8, pady=8)
        self.paned.add(self._build_form(self.paned), weight=3)
        self.paned.add(self._build_run_panel(self.paned), weight=2)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        # Sash position and Entry scrolling both need the real layout, which
        # only exists once the window is drawn.
        root.after(100, self._after_layout)

    def _after_layout(self) -> None:
        # Give the form the height it needs to show without scrolling, while
        # keeping at least ~150 px for the log.
        wanted = self.form_header.winfo_reqheight() + self.form_inner.winfo_reqheight() + 12
        self.paned.sashpos(0, max(200, min(wanted, self.paned.winfo_height() - 150)))
        # Long paths are most informative at their end (the filename), but an
        # Entry shows the start.
        self._show_path_ends()

    def _build_form(self, parent: tk.Widget) -> ttk.Frame:
        outer = ttk.Frame(parent)
        self.form_header = ttk.Label(outer, text="Fields marked * are required. Output fields (files the pipeline "
                                                 "writes) are shaded green. Hover any field for its description.")
        self.form_header.pack(anchor="w", pady=(0, 4))

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
        for group in build_parser()._action_groups:
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
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x", pady=(0, 4))
        self.run_button = ttk.Button(buttons, text="Run pipeline", command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        ttk.Button(buttons, text="Reset to defaults", command=self.reset).pack(side="left")
        ttk.Button(buttons, text="Clear log", command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=4)
        self.status = ttk.Label(buttons, text="Ready")
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

    def _show_path_ends(self) -> None:
        for _action, kind, _var, entry in self.fields:
            if kind == "path":
                entry.xview_moveto(1.0)

    def build_command(self) -> list[str]:
        """Translate the form into a run_pipeline.py command line.

        Empty optional fields are omitted, so run_pipeline.py falls back to its
        own default for them -- same as leaving the flag off on the CLI.
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
                # --gcs-rot, which run_pipeline.py treats as no rotation.
                args.append(flag)
                args.extend(var.tokens())
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
        return [sys.executable, "-u", str(PIPELINE_SCRIPT), *args]

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
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()

        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.configure(text="Running...")
        self.root.after(100, self._poll_output)

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
        self.root.after(100, self._poll_output)

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
            if kind == "rotation":
                var.reset()
            else:
                var.set(bool(action.default) if kind == "flag" else _default_text(action))
        self._show_path_ends()

    def on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askokcancel("Pipeline running", "A run is still in progress. Stop it and quit?"):
                return
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PipelineGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
