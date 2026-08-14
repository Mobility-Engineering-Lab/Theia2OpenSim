"""Generate an OpenSim InverseKinematicsTool setup file and run it via opensim-cmd.

Companion to opensim_scaling.py, but for the whole dynamic trial rather than a
single static pose: given a scaled model and a full-trial marker .trc (see
workflow_utils.write_dynamic_trc_from_c3d), runs OpenSim's IK solver frame by frame
to derive joint angles that respect the model's actual joint definitions (hinge
axes, ranges, coupling), instead of workflow_utils.build_mot_dataframe's per-segment
Euler-angle decomposition.

Marker-only IK (the original design here) turned out to be underdetermined with
markerstheia.xml's one-point-per-segment set (no torso marker at all): the
assembler would get stuck at an unconverged value for long stretches of frames and
occasionally lurch to a different one -- see project_marker_only_ik_attempt.md
memory. write_ik_setup_xml now optionally accepts a coordinate_file (typically the
analytic build_mot_dataframe motion) added as low-weight IKCoordinateTask priors --
unlike opensim_scaling.py's MarkerPlacer setup, which uses coordinates as a
dominant weight-100 prior over weight-5 markers, here the markers should still
dominate (marker_weight >> coordinate_weight) so the priors act as a regularizer
that keeps the solver near a physically sane pose rather than reproducing the
analytic angles outright.
"""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd

_IK_SETUP_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
\t<InverseKinematicsTool name="{subject_name}">
\t\t<results_directory>{results_directory}</results_directory>
\t\t<input_directory />
\t\t<model_file>{model_file}</model_file>
\t\t<constraint_weight>Inf</constraint_weight>
\t\t<accuracy>{accuracy}</accuracy>
\t\t<IKTaskSet>
\t\t\t<objects>
{ik_tasks}
\t\t\t</objects>
\t\t\t<groups />
\t\t</IKTaskSet>
\t\t<marker_file>{marker_file}</marker_file>
\t\t<coordinate_file>{coordinate_file}</coordinate_file>
\t\t<time_range> {time_range_start} {time_range_end}</time_range>
\t\t<report_errors>true</report_errors>
\t\t<output_motion_file>{output_motion_file}</output_motion_file>
\t\t<report_marker_locations>false</report_marker_locations>
\t</InverseKinematicsTool>
</OpenSimDocument>
"""

_IK_MARKER_TASK_TEMPLATE = (
    '\t\t\t\t<IKMarkerTask name="{name}">\n'
    "\t\t\t\t\t<apply>true</apply>\n"
    "\t\t\t\t\t<weight>{weight}</weight>\n"
    "\t\t\t\t</IKMarkerTask>"
)

_IK_COORDINATE_TASK_TEMPLATE = (
    '\t\t\t\t<IKCoordinateTask name="{name}">\n'
    "\t\t\t\t\t<apply>true</apply>\n"
    "\t\t\t\t\t<weight>{weight}</weight>\n"
    "\t\t\t\t\t<value_type>from_file</value_type>\n"
    "\t\t\t\t\t<value>0</value>\n"
    "\t\t\t\t</IKCoordinateTask>"
)

_IK_COORDINATE_DEFAULT_TASK_TEMPLATE = (
    '\t\t\t\t<IKCoordinateTask name="{name}">\n'
    "\t\t\t\t\t<apply>true</apply>\n"
    "\t\t\t\t\t<weight>{weight}</weight>\n"
    "\t\t\t\t\t<value_type>default_value</value_type>\n"
    "\t\t\t\t\t<value>0</value>\n"
    "\t\t\t\t</IKCoordinateTask>"
)

# gait2392 coordinates that DEFAULT_TRC_MARKER_SEGMENT_MAP's markers only weakly
# constrain (e.g. subtalar_angle sits between the ankle_angle-driven talus and the
# single RANKLE/LANKLE marker on calcn, leaving it with no independent signal) and
# that build_mot_dataframe doesn't compute/export at all, so they get no
# from_file prior either. Left fully unconstrained they wander to +/-300+ degrees
# in testing -- pin them at their default (0) with a from_file-style
# IKCoordinateTask instead, exactly like opensim_scaling.py's MarkerPlacer setup
# already does for the same coordinates and the same reason.
DEFAULT_VALUE_COORDINATES = ["subtalar_angle_r", "subtalar_angle_l"]


def write_ik_setup_xml(
    output_xml_path: str | Path,
    model_file: str | Path,
    marker_file: str | Path,
    output_motion_file: str | Path,
    time_range: tuple[float, float],
    marker_names: List[str],
    accuracy: float = 1e-5,
    subject_name: str = "theia2opensim-ik",
    marker_weight: float = 1.0,
    coordinate_file: str | Path | None = None,
    coordinate_names: List[str] | None = None,
    coordinate_weight: float = 1.0,
    default_value_coordinate_names: List[str] | None = DEFAULT_VALUE_COORDINATES,
) -> Path:
    """Write an InverseKinematicsTool setup XML.

    coordinate_file / coordinate_names / coordinate_weight are optional: when
    coordinate_file is given, an IKCoordinateTask (value_type=from_file) is added
    for each name in coordinate_names, in addition to the marker tasks -- see the
    module docstring for why (a low-weight regularizer, not a dominant prior like
    opensim_scaling.py's MarkerPlacer setup). Omit coordinate_file for pure
    marker-only IK.

    default_value_coordinate_names: coordinates pinned at their default value
    (value_type=default_value) regardless of coordinate_file -- see
    DEFAULT_VALUE_COORDINATES.

    All input/output paths are resolved to absolute, forward-slash paths so the
    XML doesn't depend on opensim-cmd's working directory. results_directory is
    set to output_motion_file's own parent directory so the report_errors marker
    error file (which OpenSim names from output_motion_file's stem, written under
    results_directory) lands somewhere deterministic -- see run_ik_tool.
    """
    output_xml_path = Path(output_xml_path)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)

    def _abs(p: str | Path) -> str:
        return Path(p).resolve().as_posix()

    output_motion_file_abs = Path(output_motion_file).resolve()
    output_motion_file_abs.parent.mkdir(parents=True, exist_ok=True)

    tasks = [_IK_MARKER_TASK_TEMPLATE.format(name=name, weight=marker_weight) for name in marker_names]
    if coordinate_file is not None:
        tasks.extend(
            _IK_COORDINATE_TASK_TEMPLATE.format(name=name, weight=coordinate_weight)
            for name in (coordinate_names or [])
        )
    tasks.extend(
        _IK_COORDINATE_DEFAULT_TASK_TEMPLATE.format(name=name, weight=coordinate_weight)
        for name in (default_value_coordinate_names or [])
    )
    ik_tasks = "\n".join(tasks)

    xml_text = _IK_SETUP_TEMPLATE.format(
        subject_name=subject_name,
        results_directory=output_motion_file_abs.parent.as_posix(),
        model_file=_abs(model_file),
        accuracy=accuracy,
        ik_tasks=ik_tasks,
        marker_file=_abs(marker_file),
        coordinate_file=_abs(coordinate_file) if coordinate_file is not None else "Unassigned",
        time_range_start=f"{time_range[0]:.6f}",
        time_range_end=f"{time_range[1]:.6f}",
        output_motion_file=output_motion_file_abs.as_posix(),
    )

    output_xml_path.write_text(xml_text, encoding="utf-8")
    return output_xml_path


class IKToolResult(NamedTuple):
    success: bool
    output_motion_file: Path
    marker_error_rms_mean: float | None
    marker_error_rms_max: float | None
    marker_error_report_path: Path | None
    stdout: str


def _parse_marker_error_report(report_path: Path) -> Tuple[float | None, float | None]:
    # OpenSim's report_errors output is a .sto Storage file: a text header ending
    # in a line containing only "endheader", then a tab-separated column-name row,
    # then one data row per frame. Column names vary slightly by OpenSim version
    # (e.g. marker_error_RMS / marker_error_max), so match on substring rather
    # than an exact name.
    try:
        lines = report_path.read_text(encoding="utf-8").splitlines()
        header_idx = next(i for i, line in enumerate(lines) if line.strip() == "endheader")
        column_names = lines[header_idx + 1].split("\t")
        data_lines = [line for line in lines[header_idx + 2 :] if line.strip()]
        data = np.array([[float(x) for x in line.split("\t")] for line in data_lines])
    except Exception:
        return None, None

    rms_col = next((i for i, name in enumerate(column_names) if "rms" in name.lower()), None)
    max_col = next((i for i, name in enumerate(column_names) if "max" in name.lower()), None)

    rms_mean = float(np.mean(data[:, rms_col])) if rms_col is not None else None
    rms_max = float(np.max(data[:, max_col])) if max_col is not None else None
    return rms_mean, rms_max


def run_ik_tool(
    setup_xml_path: str | Path,
    output_motion_file: str | Path,
    subject_name: str,
    opensim_cmd: str | Path | None = None,
) -> IKToolResult:
    """Run `opensim-cmd run-tool` on an InverseKinematicsTool setup file.

    subject_name must match the subject_name passed to write_ik_setup_xml for this
    same run: empirically (confirmed against a real opensim-cmd run), OpenSim names
    the report_errors output "<subject_name>_ik_marker_errors.sto" -- based on the
    <InverseKinematicsTool name="..."> attribute, not on output_motion_file -- and
    writes it into results_directory (output_motion_file's parent, per
    write_ik_setup_xml). Passing a run-specific subject_name (e.g. derived from
    output_motion_file's stem) keeps this deterministic when multiple trials share
    an output directory; a same-named glob is only used as a last-resort fallback.

    opensim_cmd: path to opensim-cmd(.exe). Falls back to whatever "opensim-cmd"
    resolves to on PATH if not given.
    """
    exe = str(opensim_cmd) if opensim_cmd is not None else shutil.which("opensim-cmd")
    if not exe or not Path(exe).exists():
        raise FileNotFoundError(
            "opensim-cmd executable not found. Pass --opensim-cmd (or set the "
            "OPENSIM_CMD environment variable) to point at OpenSim's bin/opensim-cmd(.exe)."
        )

    proc = subprocess.run(
        [exe, "run-tool", str(Path(setup_xml_path).resolve())],
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout + proc.stderr

    output_motion_file = Path(output_motion_file)
    success = proc.returncode == 0 and output_motion_file.exists()

    report_path = None
    error_report_candidate = output_motion_file.with_name(f"{subject_name}_ik_marker_errors.sto")
    if error_report_candidate.exists():
        report_path = error_report_candidate
    elif output_motion_file.parent.exists():
        matches = list(output_motion_file.parent.glob("*_ik_marker_errors.sto"))
        if matches:
            report_path = matches[0]

    rms_mean = rms_max = None
    if report_path is not None:
        rms_mean, rms_max = _parse_marker_error_report(report_path)

    return IKToolResult(
        success=success,
        output_motion_file=output_motion_file,
        marker_error_rms_mean=rms_mean,
        marker_error_rms_max=rms_max,
        marker_error_report_path=report_path,
        stdout=stdout,
    )


def parse_coordinate_ranges_deg(model_file: str | Path) -> Dict[str, Tuple[float, float]]:
    """
    Parse each Coordinate's declared <range> (radians) from an OpenSim .osim file,
    in degrees, keyed by coordinate name.

    Used by unwrap_ik_motion to resolve the +/-360 degree branch ambiguity IK can
    settle on for gait2392's unclamped rotational coordinates: a hinge/ball-joint
    rotation of theta is kinematically identical to theta +/- 360 degrees, so
    without clamping the assembler is free to report whichever branch it happened
    to converge to (and can get stuck there via frame-to-frame warm-starting) --
    see project_marker_only_ik_attempt.md memory. Confirmed empirically against a
    real IK run: e.g. a reported knee_angle_r of 344 degrees was the correct pose,
    just 360 degrees off from the declared -120..10 degree range.
    """
    tree = ET.parse(model_file)
    ranges: Dict[str, Tuple[float, float]] = {}
    for coord in tree.getroot().iter("Coordinate"):
        name = coord.get("name")
        range_el = coord.find("range")
        if not name or range_el is None or not range_el.text:
            continue
        lo_rad, hi_rad = (float(x) for x in range_el.text.split())
        ranges[name] = (float(np.degrees(lo_rad)), float(np.degrees(hi_rad)))
    return ranges


def unwrap_ik_motion(
    df: pd.DataFrame,
    ranges: Dict[str, Tuple[float, float]],
    skip_columns: Tuple[str, ...] = ("time", "pelvis_tx", "pelvis_ty", "pelvis_tz"),
) -> pd.DataFrame:
    """
    Bring each rotational coordinate's value to its representative closest to the
    center of that coordinate's declared range (see parse_coordinate_ranges_deg),
    by adding/subtracting whole multiples of 360 degrees, independently per frame.

    This fixes IK output that converged to a physically-correct-but-unwrapped
    +/-360-degree branch (confirmed for gait2392's knee/ankle coordinates). It does
    NOT fix a genuine alternate Euler-angle branch for multi-axis joints (a 3-axis
    rotation can have two different angle triples representing the same physical
    orientation, roughly 180 degrees apart on two of the three axes) -- observed on
    the left hip in testing, where this unwrap alone was not sufficient. See
    project_marker_only_ik_attempt.md memory.
    """
    df = df.copy()
    for column in df.columns:
        if column in skip_columns or column not in ranges:
            continue
        lo, hi = ranges[column]
        center = (lo + hi) / 2.0
        df[column] = df[column] - 360.0 * np.round((df[column] - center) / 360.0)
    return df
