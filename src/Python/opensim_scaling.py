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
# against -- is read from the static coordinate .mot. The three lumbar
# entries are dropped along with HEAD/torso when include_head=False (see
# build_scale_tool): _LUMBAR_COORDINATE_NAMES marks which ones.
_LUMBAR_COORDINATE_NAMES = frozenset({"lumbar_extension", "lumbar_bending", "lumbar_rotation"})

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
    include_head: bool = True,
) -> osim.ScaleTool:
    """Build an in-memory ScaleTool for gait2392-style measurement scaling.

    Reproduces the measurement set, IK marker tasks, and IK coordinate tasks
    in sample_data/Scaling_Setup.xml. All input/output paths are resolved to
    absolute paths so the tool doesn't depend on any process working
    directory.

    mass is required: it becomes the scaled model's total mass, and every
    downstream inverse-dynamics result scales with it.

    include_head=False drops the "torso" measurement, the HEAD IK marker
    task, and the three lumbar IK coordinate tasks, reproducing scaling as it
    behaved before HEAD-based torso scaling was added. Theia's head_4X4 is
    not always reliable -- some captures report it as a constant identity
    placeholder (never actually tracked) rather than NaN, which passes the
    required-label and gap checks silently but corrupts the torso scale
    factor and, because MarkerPlacer solves all markers jointly, distorts
    nearby joint placement too. Pass include_head=False for any subject/trial
    where head_4X4 isn't trustworthy; marker_file must then also be built
    without a HEAD marker (see the marker_segment_map argument to
    workflow_utils.write_static_trc_from_c3d).
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
        if name == "torso" and not include_head:
            continue
        measurement = osim.Measurement()
        measurement.setName(name)
        measurement.setApply(True)

        pair_set = measurement.getMarkerPairSet()
        for marker_1, marker_2 in marker_pairs:
            pair = osim.MarkerPair()
            pair.setMarkerName(0, marker_1)
            pair.setMarkerName(1, marker_2)
            # adoptAndAppend() transfers ownership of `pair` to the C++ Set,
            # but the SWIG wrapper's own `thisown` flag stays True -- so at
            # interpreter exit, Python's finalizer *also* tries to free the
            # same C++ object the Set already owns, corrupting the heap
            # (crashes with STATUS_HEAP_CORRUPTION, no traceback, after all
            # real work -- including this function's return value -- has
            # already completed). Disowning every adopted object below
            # avoids it; this applies to every adoptAndAppend call in this
            # module, including write_external_loads_xml's.
            pair_set.adoptAndAppend(pair)
            pair.thisown = False

        body_scale_set = measurement.getBodyScaleSet()
        for body_name, axes in body_scales:
            body_scale = osim.BodyScale()
            body_scale.setName(body_name)
            body_scale.setAxisNames(_array_str(list(axes)))
            body_scale_set.adoptAndAppend(body_scale)
            body_scale.thisown = False

        measurement_set.adoptAndAppend(measurement)
        measurement.thisown = False

    scaler.setMarkerFileName(_abs(marker_file))
    scaler.setTimeRange(time_range_arr)
    scaler.setPreserveMassDist(False)
    scaler.setOutputModelFileName(_abs(output_model_file))

    placer = tool.getMarkerPlacer()
    placer.setApply(True)

    ik_task_set = placer.getIKTaskSet()
    for marker_name in _IK_MARKER_TASKS:
        if marker_name == "HEAD" and not include_head:
            continue
        marker_task = osim.IKMarkerTask()
        marker_task.setName(marker_name)
        marker_task.setApply(True)
        marker_task.setWeight(_IK_MARKER_WEIGHT)
        ik_task_set.adoptAndAppend(marker_task)
        marker_task.thisown = False

    for coord_name, value_type in _IK_COORDINATE_TASKS:
        if coord_name in _LUMBAR_COORDINATE_NAMES and not include_head:
            continue
        coord_task = osim.IKCoordinateTask()
        coord_task.setName(coord_name)
        coord_task.setApply(True)
        coord_task.setWeight(_IK_COORDINATE_WEIGHT)
        coord_task.setValueType(_VALUE_TYPES[value_type])
        coord_task.setValue(0.0)
        ik_task_set.adoptAndAppend(coord_task)
        coord_task.thisown = False

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
    include_head: bool = True,
) -> ScaleToolResult:
    """Build a ScaleTool via build_scale_tool and run it through the OpenSim
    Python API.

    If setup_xml_path is given, the tool is also exported there via
    printToXML() before running -- a human-readable artifact for provenance
    or the OpenSim GUI, not something this function reloads to run.

    Marker error (RMS/max) is recovered from OpenSim's own log output,
    captured through a temporary Logger file sink rather than scraped from a
    subprocess's stdout.

    include_head is forwarded to build_scale_tool -- see its docstring.
    marker_file must be built to match (no HEAD marker) when False.
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
        include_head=include_head,
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


def write_external_loads_xml(
    output_path: str | Path,
    grf_mot_file: str | Path,
    plate_body_names: dict[int, str],
) -> Path:
    """Write an OpenSim ExternalLoads settings XML -- the file OpenSim's
    Inverse Dynamics / Inverse Kinematics tools actually consume to know
    which body each force plate's force/point/torque columns apply to.

    Built through the OpenSim Python API (osim.ExternalLoads /
    osim.ExternalForce) and serialized with printToXML(), the same pattern
    as run_scale_tool -- this is never reloaded to drive a run, only written
    out, so the ScaleTool(path).run() path-resolution bug documented on
    run_scale_tool does not apply here.

    plate_body_names: {plate_index: body_name}, e.g. {1: "calcn_r",
    2: "calcn_l"} for a gait2392-style model. See
    workflow_utils.detect_grf_plate_feet for an automatic (heuristic) guess
    at which foot each plate belongs to -- verify it before trusting it for
    Inverse Dynamics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loads = osim.ExternalLoads()
    loads.setName("external_loads")
    loads.setDataFileName(_abs(grf_mot_file))

    for plate_idx, body_name in sorted(plate_body_names.items()):
        force = osim.ExternalForce()
        force.setName(f"externalforce_{plate_idx}_{body_name}")
        force.set_applied_to_body(body_name)
        force.set_force_expressed_in_body("ground")
        force.set_point_expressed_in_body("ground")
        force.set_force_identifier(f"{plate_idx}_ground_force_v")
        force.set_point_identifier(f"{plate_idx}_ground_force_p")
        force.set_torque_identifier(f"{plate_idx}_ground_torque_")
        loads.adoptAndAppend(force)
        force.thisown = False  # see the adoptAndAppend note above

    loads.printToXML(str(output_path))
    return output_path


def write_inverse_dynamics_setup_xml(
    output_path: str | Path,
    model_file: str | Path,
    coordinates_file: str | Path,
    external_loads_file: str | Path,
    time_range: tuple[float, float],
    results_dir: str | Path,
    output_gen_force_file: str = "inverse_dynamics.sto",
) -> Path:
    """Write an OpenSim InverseDynamicsTool settings XML tying together the
    scaled model, the aligned IK coordinates, and an ExternalLoads XML (see
    write_external_loads_xml) into one file ready to open in the OpenSim GUI
    or run directly via osim.InverseDynamicsTool(path).run().

    Like write_external_loads_xml, this is only ever written here, never
    reloaded to drive a run from this codebase -- so the ScaleTool(path).run()
    path-resolution bug documented on run_scale_tool does not apply. Absolute
    paths are used throughout regardless, so the setup opens correctly no
    matter what directory it's later run from.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tool = osim.InverseDynamicsTool()
    tool.setName("inverse_dynamics")
    tool.setModelFileName(_abs(model_file))
    tool.setCoordinatesFileName(_abs(coordinates_file))
    tool.setExternalLoadsFileName(_abs(external_loads_file))
    tool.setStartTime(time_range[0])
    tool.setEndTime(time_range[1])
    tool.setLowpassCutoffFrequency(-1.0)
    tool.setResultsDir(_abs(results_dir))
    tool.setOutputGenForceFileName(output_gen_force_file)

    tool.printToXML(str(output_path))
    return output_path
