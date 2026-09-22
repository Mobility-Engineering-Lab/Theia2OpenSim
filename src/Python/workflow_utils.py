"""Core utilities for converting Theia3D C3D rotations into OpenSim-ready outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from ezc3d import c3d
from pyomeca import Angles, Rototrans

REQUIRED_ROTATION_LABELS = [
    "pelvis_4X4",
    "torso_4X4",
    "head_4X4",
    "l_thigh_4X4",
    "l_shank_4X4",
    "l_foot_4X4",
    "r_thigh_4X4",
    "r_shank_4X4",
    "r_foot_4X4",
    "r_toes_4X4",
    "l_toes_4X4",
]


class WorkflowValidationError(RuntimeError):
    """Raised when required input data for the workflow is missing or invalid."""

DEFAULT_GCS_ROTATIONS = (
    ("Z", -90.0),
    ("X", -90.0),
)


def _axis_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """
    Return a 3x3 rotation matrix about X, Y, or Z.

    Positive angles follow the right-hand rule.
    """
    axis = axis.upper()

    if axis not in {"X", "Y", "Z"}:
        raise WorkflowValidationError(
            f"Invalid rotation axis '{axis}'. Use X, Y, or Z."
        )

    theta = np.deg2rad(float(angle_deg))
    c = np.cos(theta)
    s = np.sin(theta)

    if axis == "X":
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0,   c,  -s],
            [0.0,   s,   c],
        ])

    if axis == "Y":
        return np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c],
        ])

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def build_gcs_to_opensim_rotation(
    rotations: Iterable[Tuple[str, float]],
) -> np.ndarray:
    """
    Construct the rotation matrix transforming coordinates from the
    source laboratory GCS into the OpenSim GCS.

    Rotations are active, fixed-axis rotations applied in the order
    supplied. Positive angles follow the right-hand rule.

    Example:
        [("Z", -90), ("X", -90)]

    applies:
        1. -90 degrees about Z
        2. -90 degrees about X

    giving:
        R_gcs_to_opensim = Rx(-90) @ Rz(-90)

    For a column vector:
        v_OpenSim = R_gcs_to_opensim @ v_source
    """
    rotations = list(rotations)

    if len(rotations) > 3:
        raise WorkflowValidationError(
            "A maximum of three GCS rotations may be specified."
        )

    R = np.eye(3)

    for axis, angle_deg in rotations:
        R_step = _axis_rotation_matrix(axis, angle_deg)

        # Rotations are applied in the order entered by the user.
        R = R_step @ R

    # Validate that the final matrix is a proper rotation matrix.
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8):
        raise WorkflowValidationError(
            "Generated GCS transformation is not orthonormal."
        )

    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-8):
        raise WorkflowValidationError(
            "Generated GCS transformation is not a proper rotation."
        )

    return R


def parse_axis_angle_rotations(
    rotation_specs: Iterable[str],
) -> List[Tuple[str, float]]:
    """
    Convert strings such as:
        ["Z:-90", "X:-90"]

    into:
        [("Z", -90.0), ("X", -90.0)]
    """
    rotations: List[Tuple[str, float]] = []

    for spec in rotation_specs:
        try:
            axis, angle = spec.split(":", 1)
            axis = axis.strip().upper()
            angle_deg = float(angle)
        except ValueError as exc:
            raise WorkflowValidationError(
                f"Invalid GCS rotation '{spec}'. "
                "Use AXIS:ANGLE, for example Z:-90."
            ) from exc

        if axis not in {"X", "Y", "Z"}:
            raise WorkflowValidationError(
                f"Invalid rotation axis '{axis}'. Use X, Y, or Z."
            )

        rotations.append((axis, angle_deg))

    if len(rotations) > 3:
        raise WorkflowValidationError(
            "A maximum of three GCS rotations may be specified."
        )

    return rotations


def load_theia_c3d(c3d_path: str | Path) -> Dict:
    """Load a Theia3D-exported C3D file."""
    return c3d(str(c3d_path))


def get_rotation_labels(c3d_obj: Dict) -> List[str]:
    return list(c3d_obj["parameters"]["ROTATION"]["LABELS"]["value"])


def find_missing_labels(labels: Iterable[str], required: Iterable[str]) -> List[str]:
    labels_set = set(labels)
    return [name for name in required if name not in labels_set]


def get_frame_rate_and_count(c3d_obj: Dict) -> Tuple[float, int]:
    # Theia stores these values in the POINT section of the C3D metadata.
    frame_rate = float(np.asarray(c3d_obj["parameters"]["POINT"]["RATE"]["value"]).squeeze())
    total_frames = int(np.asarray(c3d_obj["parameters"]["POINT"]["FRAMES"]["value"]).squeeze())
    return frame_rate, total_frames


def _rotation_data_transposed(c3d_obj: Dict) -> np.ndarray:
    # Move the segment index to the front so each 4x4 pose matrix is easy to slice.
    rotation_data = c3d_obj["data"]["rotations"]
    return np.transpose(rotation_data, (2, 0, 1, 3))


def _translation_scale_to_mm(c3d_obj: Dict) -> float:
    # Segment pose translations follow whatever length unit POINT:UNITS declares
    # (Theia doesn't expose a separate ROTATION:UNITS). Some exports are 'mm',
    # others are 'm' -- the rest of this module assumes millimeters throughout
    # (matches the .trc convention and the mm->m divisions in build_mot_dataframe),
    # so normalize to mm here, once, at the source.
    units = c3d_obj["parameters"]["POINT"]["UNITS"]["value"]
    unit = str(units[0]).strip().lower() if len(units) else "mm"
    if unit == "mm":
        return 1.0
    if unit == "m":
        return 1000.0
    raise WorkflowValidationError(f"Unsupported POINT:UNITS '{unit}' (expected 'mm' or 'm').")


def get_segment_pose(c3d_obj: Dict, segment_label: str) -> np.ndarray:
    labels = get_rotation_labels(c3d_obj)
    if segment_label not in labels:
        raise WorkflowValidationError(f"Missing segment label: {segment_label}")

    # Reorder the rotation array so we can access one segment at a time.
    rotation_data = _rotation_data_transposed(c3d_obj)
    pose = rotation_data[labels.index(segment_label)]

    # Replace NaNs with zeros and force a valid homogeneous transform row.
    pose = np.nan_to_num(pose, nan=0.0)
    pose[3, :, :] = np.array([0.0, 0.0, 0.0, 1.0])[:, np.newaxis]
    pose[:3, 3, :] *= _translation_scale_to_mm(c3d_obj)
    return pose


def _ensure_degrees(values: np.ndarray) -> np.ndarray:
    # pyomeca may return radians; convert only when the values look like angles in radians.
    values = np.asarray(values, dtype=float)
    max_abs = np.nanmax(np.abs(values)) if values.size else 0.0
    if max_abs <= (2.0 * np.pi + 1.0):
        return np.degrees(values)
    return values


def _invert_rigid_pose(pose: np.ndarray) -> np.ndarray:
    # Manual rigid-transform inverse (R^T, -R^T @ t), computed directly with numpy.
    # pyomeca's Rototrans.from_transposed_rototrans does NOT correctly invert a
    # Rototrans built from a raw numpy array -- verified against a known 90-degree
    # rotation: composing its "inverse" with the original does not yield the
    # identity, and the rotation submatrix it returns is left completely
    # unchanged (not transposed at all). This bypasses that broken path.
    rotation = pose[:3, :3, :]
    translation = pose[:3, 3, :]
    inv_pose = np.zeros_like(pose)
    inv_rotation = np.transpose(rotation, (1, 0, 2))
    inv_pose[:3, :3, :] = inv_rotation
    inv_pose[:3, 3, :] = -np.einsum("ijt, jt -> it", inv_rotation, translation)
    inv_pose[3, 3, :] = 1.0
    return inv_pose


def absolute_segment_angles(segment_pose: np.ndarray, sequence: str = "xyz") -> np.ndarray:
    # Convert the absolute segment pose into Euler angles for reporting.
    segment_rt_t = Rototrans(_invert_rigid_pose(segment_pose))
    angles = np.asarray(Angles.from_rototrans(segment_rt_t, sequence))
    angles = _ensure_degrees(angles)
    # Euler decomposition wraps at +/-180 degrees per axis (an atan2 branch cut).
    # When the true absolute orientation sits near that boundary, a sub-degree
    # frame-to-frame change can flip the extracted angle by ~360 degrees even
    # though nothing moved. Unwrapping restores continuity within the trial --
    # it does not change the reference frame or re-zero the angle, it only
    # removes this discontinuity artifact.
    #
    # Frames with no real tracking data get a degenerate (all-zero) rotation
    # submatrix from get_segment_pose's nan_to_num. These aren't real motion and
    # must be excluded from the unwrap chain, or the untracked frames can drift
    # by a spurious +/-360 just to stay "continuous" with real motion elsewhere
    # in the trial that crossed the wrap boundary.
    return _unwrap_degrees(angles, _valid_pose_mask(segment_pose), axis=-1)


def _unwrap_degrees(angles: np.ndarray, valid: np.ndarray, axis: int = -1) -> np.ndarray:
    # Unwrap only the valid (actually-tracked) frames, in their original order,
    # and leave invalid/gap frames exactly as computed -- see absolute_segment_angles.
    angles = np.moveaxis(np.array(angles, dtype=float, copy=True), axis, -1)
    if valid.any():
        for row in np.ndindex(angles.shape[:-1]):
            angles[row][valid] = np.unwrap(angles[row][valid], period=360.0)
    return np.moveaxis(angles, -1, axis)


def _valid_pose_mask(pose: np.ndarray) -> np.ndarray:
    # Frames with no real tracking data get a degenerate (all-zero) rotation
    # submatrix from get_segment_pose's nan_to_num -- see absolute_segment_angles.
    return np.linalg.norm(pose[:3, :3, :], axis=(0, 1)) > 1e-9


def relative_segment_pose(parent_pose: np.ndarray, child_pose: np.ndarray) -> np.ndarray:
    # Full 4x4 pose of the child expressed in the parent's frame (parent^-1 @ child):
    # rotation R_parent^T @ R_child, translation R_parent^T @ (t_child - t_parent).
    parent_rt_t = _invert_rigid_pose(parent_pose)
    return np.asarray(np.einsum("ijt, jkt -> ikt", parent_rt_t, child_pose))


def relative_segment_angles(parent_pose: np.ndarray, child_pose: np.ndarray, sequence: str = "xyz") -> np.ndarray:
    # Compute the child pose relative to the parent pose, then extract joint angles.
    rel_rt = relative_segment_pose(parent_pose, child_pose)
    angles = np.asarray(Angles.from_rototrans(Rototrans(rel_rt), sequence))
    angles = _ensure_degrees(angles)
    valid = _valid_pose_mask(parent_pose) & _valid_pose_mask(child_pose)
    return _unwrap_degrees(angles, valid, axis=-1)


def build_mot_dataframe(c3d_obj: Dict,gcs_to_opensim_R: np.ndarray | None = None,) -> pd.DataFrame:
    # Build the OpenSim motion table one signal block at a time.

    frame_rate, total_frames = get_frame_rate_and_count(c3d_obj)
    frame_values = np.arange(total_frames)
    time = np.round(frame_values / frame_rate, 5)

    # Pull the segment poses used throughout the lower-body and trunk calculations.
    pelvis = get_segment_pose(c3d_obj, "pelvis_4X4")
    torso = get_segment_pose(c3d_obj, "torso_4X4")

    l_thigh = get_segment_pose(c3d_obj, "l_thigh_4X4")
    l_shank = get_segment_pose(c3d_obj, "l_shank_4X4")
    l_foot = get_segment_pose(c3d_obj, "l_foot_4X4")

    r_thigh = get_segment_pose(c3d_obj, "r_thigh_4X4")
    r_shank = get_segment_pose(c3d_obj, "r_shank_4X4")
    r_foot = get_segment_pose(c3d_obj, "r_foot_4X4")

    # ---------------------------------------------------------------
    # Source laboratory GCS -> OpenSim GCS
    # ---------------------------------------------------------------
    if gcs_to_opensim_R is None:
        gcs_to_opensim_R = build_gcs_to_opensim_rotation(
            DEFAULT_GCS_ROTATIONS
        )

    # Transform pelvis orientation:
    # R_pelvis^OpenSim =
    #     R_GCS^OpenSim @ R_pelvis^source
    #
    # Transform pelvis position:
    # r_pelvis^OpenSim =
    #     R_GCS^OpenSim @ r_pelvis^source
    
    # ---------------------------------------------------------------
    # Fixed Theia pelvis-local -> OpenSim gait2392 pelvis-local mapping
    # ---------------------------------------------------------------
    pelvis_local_to_opensim_R = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    pelvis_os = np.zeros_like(pelvis)

    # Pelvis orientation:
    #
    # R_pelvis^OpenSim =
    #     R_GCS^OpenSim
    #     @ R_pelvis^Theia
    #     @ R_local
    #
    # Left side:
    #   laboratory GCS -> OpenSim GCS
    #
    # Right side:
    #   Theia pelvis local basis -> gait2392 pelvis local basis
    pelvis_os[:3, :3, :] = np.einsum(
        "ij,jkt,kl->ilt",
        gcs_to_opensim_R,
        pelvis[:3, :3, :],
        pelvis_local_to_opensim_R,
    )

    # Pelvis position is a point in the laboratory GCS, so it only
    # requires the global coordinate-system transformation.
    pelvis_os[:3, 3, :] = (
        gcs_to_opensim_R @ pelvis[:3, 3, :]
    )

    pelvis_os[3, 3, :] = 1.0


    identity_ref = np.repeat(
    np.eye(4)[:, :, None],
    pelvis_os.shape[2],
    axis=2,
)

    # Pelvis angles are used as the OpenSim pelvis columns. Sequence must be
    # 'zxy' (intrinsic Z -> X -> Y), matching gait2392's ground_pelvis.
    pelvis_angles = relative_segment_angles(
        identity_ref,
        pelvis_os,
        sequence="zxy",
)

    pelvis_tx_source = pelvis_os[0, 3, :]
    pelvis_tz_source = pelvis_os[2, 3, :]
   
    # Lumbar motion is the torso relative to the pelvis.
    lumbar_angles = relative_segment_angles(pelvis, torso, sequence="zxy")

    # Joint angles are calculated from child segment motion relative to its parent segment.
    knee_angles_l = relative_segment_angles(l_thigh, l_shank)
    knee_angles_r = relative_segment_angles(r_thigh, r_shank)

    hip_angles_l = relative_segment_angles(pelvis, l_thigh)
    hip_angles_r = relative_segment_angles(pelvis, r_thigh)

    ankle_angles_l = relative_segment_angles(l_shank, l_foot)
    ankle_angles_r = relative_segment_angles(r_shank, r_foot)

    # Match the notebook's output columns so the exported MOT stays familiar.
    data = {
        "time": time,
        "pelvis_tilt": pelvis_angles[0, 0, :],
        "pelvis_list": pelvis_angles[1, 0, :],
        "pelvis_rotation": pelvis_angles[2, 0, :],
        "hip_flexion_r": hip_angles_r[0, 0, :],
        "hip_adduction_r":  hip_angles_r[1, 0, :],
        "hip_rotation_r":  hip_angles_r[2, 0, :],
        "knee_angle_r": knee_angles_r[0, 0, :],
        "ankle_angle_r": ankle_angles_r[0, 0, :],
        "hip_flexion_l": hip_angles_l[0, 0, :],
        "hip_adduction_l": -hip_angles_l[1, 0, :],
        "hip_rotation_l":  -hip_angles_l[2, 0, :],
        "knee_angle_l": knee_angles_l[0, 0, :],
        "ankle_angle_l": ankle_angles_l[0, 0, :],
        # Pelvis translations: already expressed in the OpenSim ground frame by
        # lab_to_opensim_R above
        "pelvis_tx": pelvis_tx_source / 1000.0,
        "pelvis_ty": pelvis_os[1, 3, :] / 1000.0,
        "pelvis_tz": pelvis_tz_source / 1000.0,
        "lumbar_bending": lumbar_angles[0, 0, :],
        "lumbar_rotation": lumbar_angles[2, 0, :],
        "lumbar_extension": lumbar_angles[1, 0, :],
    }

    return pd.DataFrame(data)


def validate_mot_dataframe(df: pd.DataFrame) -> List[str]:
    # Keep validation lightweight: empty data, missing time, non-finite values, and time order.
    issues: List[str] = []
    if df.empty:
        issues.append("DataFrame is empty.")
        return issues

    if "time" not in df.columns:
        issues.append("Missing 'time' column.")
        return issues

    if not np.all(np.isfinite(df.to_numpy(dtype=float, copy=True))):
        issues.append("DataFrame contains non-finite values (NaN or inf).")

    time_values = df["time"].to_numpy(dtype=float)
    if np.any(np.diff(time_values) < 0):
        issues.append("Time values are not monotonic increasing.")

    return issues


def trim_edges_zeros_and_reset_time(signal: np.ndarray, frame_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    # Trim leading/trailing zero regions so the exported motion starts and ends on movement.
    signal = np.asarray(signal).flatten()
    non_zero_indices = np.nonzero(signal)[0]
    if non_zero_indices.size == 0:
        return np.array([]), np.array([])

    start_idx = non_zero_indices[0]
    end_idx = non_zero_indices[-1] + 1
    trimmed_signal = signal[start_idx:end_idx]
    time_trimmed = np.arange(len(trimmed_signal)) / frame_rate
    return trimmed_signal, time_trimmed


def trim_dataframe(df: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    # Trim each signal independently, then re-align everything to the shortest valid span.
    trimmed_data: Dict[str, np.ndarray] = {}
    reference_time: np.ndarray | None = None

    for col in df.columns:
        if col == "time":
            continue
        trimmed_signal, trimmed_time = trim_edges_zeros_and_reset_time(df[col].to_numpy(), frame_rate)
        if trimmed_signal.size == 0:
            continue
        trimmed_data[col] = trimmed_signal
        if reference_time is None or len(trimmed_time) < len(reference_time):
            reference_time = trimmed_time

    if not trimmed_data or reference_time is None or len(reference_time) == 0:
        raise WorkflowValidationError("Unable to trim data: all signals are zero or empty.")

    # Shorten every signal to the same trimmed duration so the MOT stays rectangular.
    for col in list(trimmed_data.keys()):
        trimmed_data[col] = trimmed_data[col][: len(reference_time)]

    ordered_columns = ["time"] + [name for name in df.columns if name != "time" and name in trimmed_data]
    out = {"time": reference_time}
    out.update({col: trimmed_data[col] for col in ordered_columns if col != "time"})
    return pd.DataFrame(out)

# Marker names must match the virtual markers defined in the OpenSim scaling marker set.
# These are not anatomical skin markers. They are virtual markers derived from Theia3D
# segment origins / joint-centre proxy locations.
DEFAULT_TRC_MARKER_SEGMENT_MAP = {
    "RASIS": "r_thigh_4X4",
    "LASIS": "l_thigh_4X4",
    "RKNEE": "r_shank_4X4",
    "LKNEE": "l_shank_4X4",
    "RANKLE": "r_foot_4X4",
    "LANKLE": "l_foot_4X4",
    "RTOE": "r_toes_4X4",
    "LTOE": "l_toes_4X4",
    "PELVIS": "pelvis_4X4",
    # HEAD drives the torso measurement. Theia's torso_4X4 origin sits at
    # ~C7, only ~94 mm below head_4X4, so a TORSO-HEAD pair is far too short
    # to derive a stable scale factor from. The PELVIS-HEAD span (~523 mm) is
    # used instead -- see the "torso" Measurement in opensim_scaling.
    "HEAD": "head_4X4",
}


def parse_frame_indices(frame_spec: str) -> np.ndarray:
    """
    Parse selected static frames.

    Accepted formats:
    - "300"
    - "290:310"       inclusive start, exclusive end
    - "290,300,310"
    """
    frame_spec = str(frame_spec).strip()

    if ":" in frame_spec:
        start, end = frame_spec.split(":")
        return np.arange(int(start), int(end), dtype=int)

    if "," in frame_spec:
        return np.asarray([int(x.strip()) for x in frame_spec.split(",")], dtype=int)

    return np.asarray([int(frame_spec)], dtype=int)


def extract_virtual_marker_positions(
    c3d_obj: Dict,
    frame_indices: Iterable[int],
    marker_segment_map: Dict[str, str] | None = None,
    gcs_to_opensim_R: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """    
    Extract virtual-marker positions from Theia3D segment origins and
    express them in the OpenSim GCS.

    The segment origins are obtained from the translational component
    of each 4x4 pose matrix and are retained in millimetres for TRC export.
  
    """
    if gcs_to_opensim_R is None:
        gcs_to_opensim_R = build_gcs_to_opensim_rotation(
            DEFAULT_GCS_ROTATIONS
        )

    if marker_segment_map is None:
        marker_segment_map = DEFAULT_TRC_MARKER_SEGMENT_MAP

    frames = np.asarray(list(frame_indices), dtype=int)

    if frames.size == 0:
        raise WorkflowValidationError(
            "No static frames were selected for TRC generation."
        )

    _, total_frames = get_frame_rate_and_count(c3d_obj)

    if np.any(frames < 0) or np.any(frames >= total_frames):
        raise WorkflowValidationError(
            f"Selected static frame(s) out of range. "
            f"Valid range is 0 to {total_frames - 1}."
        )

    marker_positions: Dict[str, np.ndarray] = {}

    for marker_name, segment_label in marker_segment_map.items():
        pose = get_segment_pose(c3d_obj, segment_label)

        # Segment origin expressed in the source laboratory GCS.
        origin = pose[:3, 3, :]

        # Source GCS -> OpenSim GCS.
        mapped_origin = (
            gcs_to_opensim_R @ origin[:, frames]
        )

        marker_positions[marker_name] = np.mean(
            mapped_origin,
            axis=1,
        )

    return marker_positions


def write_static_trc(
    marker_positions: Dict[str, np.ndarray],
    output_path: str | Path,
    data_rate: float,
    repeat_frames: int = 6,
    units: str = "mm",
) -> Path:
    """
    Write a static OpenSim .trc file using repeated virtual-marker positions.

    The same averaged static marker positions are repeated across several frames
    to mimic a static calibration trial for the OpenSim Scale Tool.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    marker_names = list(marker_positions.keys())
    num_markers = len(marker_names)
    times = np.arange(repeat_frames, dtype=float) / float(data_rate)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write(f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}\n")
        file.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
            "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )
        file.write(
            f"{data_rate:.6f}\t{data_rate:.6f}\t{repeat_frames}\t{num_markers}\t"
            f"{units}\t{data_rate:.6f}\t1\t{repeat_frames}\n"
        )

        marker_header = ["Frame#", "Time"]
        for marker_name in marker_names:
            marker_header.extend([marker_name, "", ""])
        file.write("\t".join(marker_header) + "\n")

        xyz_header = ["", ""]
        for idx in range(1, num_markers + 1):
            xyz_header.extend([f"X{idx}", f"Y{idx}", f"Z{idx}"])
        file.write("\t".join(xyz_header) + "\n")

        for frame_idx, time_value in enumerate(times, start=1):
            row = [str(frame_idx), f"{time_value:.6f}"]
            for marker_name in marker_names:
                x, y, z = marker_positions[marker_name]
                row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
            file.write("\t".join(row) + "\n")

    return output_path


def write_static_trc_from_c3d(
    c3d_obj: Dict,
    output_path: str | Path,
    frame_indices: Iterable[int],
    repeat_frames: int = 6,
    marker_segment_map: Dict[str, str] | None = None,
    gcs_to_opensim_R: np.ndarray | None = None,
) -> Path:
    """
    Build and write a static .trc file directly from a Theia3D C3D file.
    """
    frame_rate, _ = get_frame_rate_and_count(c3d_obj)
    marker_positions = extract_virtual_marker_positions(
        c3d_obj=c3d_obj,
        frame_indices=frame_indices,
        marker_segment_map=marker_segment_map,
        gcs_to_opensim_R=gcs_to_opensim_R,
    )

    return write_static_trc(
        marker_positions=marker_positions,
        output_path=output_path,
        data_rate=frame_rate,
        repeat_frames=repeat_frames,
        units="mm",
    )


def write_mot(df: pd.DataFrame, output_path: str | Path) -> Path:
    # Write the OpenSim .mot header followed by a tab-delimited table.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = float(df["time"].iloc[0])
    end_time = float(df["time"].iloc[-1])
    # OpenSim expects a short header before the tabular motion data.
    header_lines = [
        output_path.name,
        "version=1",
        f"datacolumns {df.shape[1]}",
        f"datarows {df.shape[0]}",
        f"range {start_time:.5f} {end_time:.5f}",
        "inDegrees=yes",
        "endheader",
    ]

    with output_path.open("w", encoding="utf-8") as file:
        for line in header_lines:
            file.write(line + "\n")
        # Column names are written on the first data line, matching the MOT text format.
        file.write("\t".join(df.columns) + "\n")
        for row in df.itertuples(index=False, name=None):
            # Format each numeric value consistently so OpenSim reads the file cleanly.
            file.write("\t".join(map(lambda x: f"{float(x):.6f}", row)) + "\n")

    return output_path