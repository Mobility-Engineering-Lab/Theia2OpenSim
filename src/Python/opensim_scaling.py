"""Build and run an OpenSim ScaleTool through the OpenSim Python API.

This mirrors the same measurement-based scaling + MarkerPlacer setup
validated by hand against TheiaTestData (see sample_data/Scaling_Setup.xml):
measurement-based scaling using the DEFAULT_TRC_MARKER_SEGMENT_MAP marker
names, followed by MarkerPlacer using the same static TRC + a self-consistent
coordinate .mot (both must be built with the same lab_to_opensim_R, i.e. share
the same reference frame, or MarkerPlacer's IK reports a large, spurious
marker error).

The ScaleTool is built directly through API calls (Measurement, MarkerPair,
BodyScale, IKMarkerTask, IKCoordinateTask, ...) and run in-memory, rather
than written to a setup XML and reloaded via `osim.ScaleTool(path)`. That
reload path has a real resolution bug in OpenSim 4.5.2: GenericModelMaker's
marker_set_file and ModelScaler's marker_file get the setup XML's own
directory re-prepended during run(), even though the stored value is already
absolute -- confirmed by reading the properties right after construction
(correct) versus after run() (silently corrupted, "Unable to open marker
file"). This does not affect opensim-cmd, only `ScaleTool(path)` loaded from
Python. Building the tool via setters and calling run() on that same object
sidesteps the bug entirely. printToXML() is still used afterward, purely to
leave a human-readable setup file behind for provenance and for opening in
the OpenSim GUI -- it is never re-loaded to drive a run.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import NamedTuple, Sequence

import opensim as osim

# Marker-pair measurements for gait2392-style measurement-based scaling.
# Each entry: (measurement name, marker pairs, body scales). HEAD scales the
# torso off the PELVIS-HEAD span -- see workflow_utils.DEFAULT_TRC_MARKER_SEGMENT_MAP
# for why TORSO-HEAD (~94 mm) is too short to use directly. foot r/l each
# scale their own side's talus; scaling talus_l off the right foot pair was a
# past bug, fixed here.
_MEASUREMENTS: tuple[tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]], ...] = (
    ("pelvis",  (("LASIS", "RASIS"),),  (("pelvis", "XYZ"),)),
    ("torso",   (("PELVIS", "HEAD"),),  (("torso", "XYZ"),)),
    ("femur r", (("RASIS", "RKNEE"),),  (("femur_r", "XYZ"),)),
    ("femur l", (("LASIS", "LKNEE"),),  (("femur_l", "XYZ"),)),
    ("tibia r", (("RKNEE", "RANKLE"),), (("tibia_r", "XYZ"),)),
    ("tibia l", (("LKNEE", "LANKLE"),), (("tibia_l", "XYZ"),)),
    ("foot r",  (("RANKLE", "RTOE"),),  (("talus_r", "XYZ"), ("calcn_r", "XYZ"), ("toes_r", "XYZ"))),
    ("foot l",  (("LANKLE", "LTOE"),),  (("talus_l", "XYZ"), ("calcn_l", "XYZ"), ("toes_l", "XYZ"))),
)

_IK_MARKER_TASKS: tuple[str, ...] = (
    "LASIS", "RASIS", "RKNEE", "LKNEE", "RTOE", "LANKLE", "LTOE", "RANKLE", "PELVIS", "HEAD",
)
_IK_MARKER_WEIGHT = 5.0

# (coordinate name, value_type). Weight is uniformly 100 for every coordinate
# task. subtalar_angle_r/l have no reliable Theia-derived value, so they stay
# at their model default rather than being driven from the coordinate file;
# every other coordinate -- including the three lumbar ones, which need a
# task now that HEAD gives MarkerPlacer something to solve the torso
# against -- is read from the static coordinate .mot.
_IK_COORDINATE_TASKS: tuple[tuple[str, str], ...] = (
    ("pelvis_tilt", "from_file"),
    ("pelvis_list", "from_file"),
    ("pelvis_rotation", "from_file"),
    ("hip_flexion_r", "from_file"),
    ("hip_adduction_r", "from_file"),
    ("hip_rotation_r", "from_file"),
    ("knee_angle_r", "from_file"),
    ("ankle_angle_r", "from_file"),
    ("subtalar_angle_r", "default_value"),
    ("hip_flexion_l", "from_file"),
    ("hip_adduction_l", "from_file"),
    ("hip_rotation_l", "from_file"),
    ("knee_angle_l", "from_file"),
    ("ankle_angle_l", "from_file"),
    ("subtalar_angle_l", "default_value"),
    ("lumbar_extension", "from_file"),
    ("lumbar_bending", "from_file"),
    ("lumbar_rotation", "from_file"),
)
_IK_COORDINATE_WEIGHT = 100.0

_VALUE_TYPES = {
    "default_value": osim.IKCoordinateTask.DefaultValue,
    "manual_value": osim.IKCoordinateTask.ManualValue,
    "from_file": osim.IKCoordinateTask.FromFile,
}

_MARKER_ERROR_RE = re.compile(
    r"marker error:\s*RMS\s*=\s*([\d.eE+-]+),\s*max\s*=\s*([\d.eE+-]+)\s*\(([^)]+)\)"
)


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def _array_str(values: Sequence[str]) -> osim.ArrayStr:
    arr = osim.ArrayStr()
    for value in values:
        arr.append(value)
    return arr


def _array_double(values: Sequence[float]) -> osim.ArrayDouble:
    arr = osim.ArrayDouble()
    for value in values:
        arr.append(value)
    return arr


def build_scale_tool(
    model_file: str | Path,
    marker_set_file: str | Path,
    marker_file: str | Path,
    coordinate_file: str | Path,
    output_model_file: str | Path,
    time_range: tuple[float, float],
    mass: float,
    subject_name: str = "theia2opensim-scaled",
    output_motion_file: str | Path | None = None,
) -> osim.ScaleTool:
    """Build an in-memory ScaleTool for gait2392-style measurement scaling.

    Reproduces the measurement set, IK marker tasks, and IK coordinate tasks
    in sample_data/Scaling_Setup.xml. All input/output paths are resolved to
    absolute paths so the tool doesn't depend on any process working
    directory.

    mass is required: it becomes the scaled model's total mass, and every
    downstream inverse-dynamics result scales with it.
    """
    if not mass > 0:
        raise ValueError(f"Subject mass must be positive, got {mass!r} kg.")

    if output_motion_file is None:
        output_motion_file = Path(output_model_file).with_name(
            Path(output_model_file).stem + "_static_ik.mot"
        )

    time_range_arr = _array_double(time_range)

    tool = osim.ScaleTool()
    tool.setName(subject_name)
    tool.setSubjectMass(mass)

    generic_model_maker = tool.getGenericModelMaker()
    generic_model_maker.setModelFileName(_abs(model_file))
    generic_model_maker.setMarkerSetFileName(_abs(marker_set_file))

    scaler = tool.getModelScaler()
    scaler.setApply(True)
    scaler.setScalingOrder(_array_str(["measurements"]))

    measurement_set = scaler.getMeasurementSet()
    for name, marker_pairs, body_scales in _MEASUREMENTS:
        measurement = osim.Measurement()
        measurement.setName(name)
        measurement.setApply(True)

        pair_set = measurement.getMarkerPairSet()
        for marker_1, marker_2 in marker_pairs:
            pair = osim.MarkerPair()
            pair.setMarkerName(0, marker_1)
            pair.setMarkerName(1, marker_2)
            pair_set.adoptAndAppend(pair)

        body_scale_set = measurement.getBodyScaleSet()
        for body_name, axes in body_scales:
            body_scale = osim.BodyScale()
            body_scale.setName(body_name)
            body_scale.setAxisNames(_array_str(list(axes)))
            body_scale_set.adoptAndAppend(body_scale)

        measurement_set.adoptAndAppend(measurement)

    scaler.setMarkerFileName(_abs(marker_file))
    scaler.setTimeRange(time_range_arr)
    scaler.setPreserveMassDist(False)
    scaler.setOutputModelFileName(_abs(output_model_file))

    placer = tool.getMarkerPlacer()
    placer.setApply(True)

    ik_task_set = placer.getIKTaskSet()
    for marker_name in _IK_MARKER_TASKS:
        marker_task = osim.IKMarkerTask()
        marker_task.setName(marker_name)
        marker_task.setApply(True)
        marker_task.setWeight(_IK_MARKER_WEIGHT)
        ik_task_set.adoptAndAppend(marker_task)

    for coord_name, value_type in _IK_COORDINATE_TASKS:
        coord_task = osim.IKCoordinateTask()
        coord_task.setName(coord_name)
        coord_task.setApply(True)
        coord_task.setWeight(_IK_COORDINATE_WEIGHT)
        coord_task.setValueType(_VALUE_TYPES[value_type])
        coord_task.setValue(0.0)
        ik_task_set.adoptAndAppend(coord_task)

    placer.setMarkerFileName(_abs(marker_file))
    placer.setCoordinateFileName(_abs(coordinate_file))
    placer.setTimeRange(time_range_arr)
    placer.setOutputMotionFileName(_abs(output_motion_file))
    placer.setOutputModelFileName(_abs(output_model_file))
    placer.setMaxMarkerMovement(-1.0)

    return tool


class ScaleToolResult(NamedTuple):
    success: bool
    output_model_file: Path
    marker_rms: float | None
    marker_max: float | None
    marker_max_name: str | None
    log_text: str


def run_scale_tool(
    model_file: str | Path,
    marker_set_file: str | Path,
    marker_file: str | Path,
    coordinate_file: str | Path,
    output_model_file: str | Path,
    time_range: tuple[float, float],
    mass: float,
    subject_name: str = "theia2opensim-scaled",
    output_motion_file: str | Path | None = None,
    setup_xml_path: str | Path | None = None,
) -> ScaleToolResult:
    """Build a ScaleTool via build_scale_tool and run it through the OpenSim
    Python API.

    If setup_xml_path is given, the tool is also exported there via
    printToXML() before running -- a human-readable artifact for provenance
    or the OpenSim GUI, not something this function reloads to run.

    Marker error (RMS/max) is recovered from OpenSim's own log output,
    captured through a temporary Logger file sink rather than scraped from a
    subprocess's stdout.
    """
    output_model_file = Path(output_model_file)

    tool = build_scale_tool(
        model_file=model_file,
        marker_set_file=marker_set_file,
        marker_file=marker_file,
        coordinate_file=coordinate_file,
        output_model_file=output_model_file,
        time_range=time_range,
        mass=mass,
        subject_name=subject_name,
        output_motion_file=output_motion_file,
    )

    if setup_xml_path is not None:
        setup_xml_path = Path(setup_xml_path)
        setup_xml_path.parent.mkdir(parents=True, exist_ok=True)
        tool.printToXML(str(setup_xml_path))

    log_fd, log_path_str = tempfile.mkstemp(suffix=".log", prefix="opensim_scale_")
    log_path = Path(log_path_str)
    os.close(log_fd)
    log_path.unlink()  # Logger creates its own file; only the unique name is needed.

    osim.Logger.addFileSink(str(log_path))
    try:
        tool.run()
    finally:
        osim.Logger.removeFileSink()

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    log_path.unlink(missing_ok=True)

    rms = marker_max = None
    max_name = None
    match = _MARKER_ERROR_RE.search(log_text)
    if match:
        rms = float(match.group(1))
        marker_max = float(match.group(2))
        max_name = match.group(3)

    success = output_model_file.exists()
    return ScaleToolResult(
        success=success,
        output_model_file=output_model_file,
        marker_rms=rms,
        marker_max=marker_max,
        marker_max_name=max_name,
        log_text=log_text,
    )
