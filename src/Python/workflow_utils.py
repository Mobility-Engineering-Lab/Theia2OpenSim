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
    return pose


def _ensure_degrees(values: np.ndarray) -> np.ndarray:
    # pyomeca may return radians; convert only when the values look like angles in radians.
    values = np.asarray(values, dtype=float)
    max_abs = np.nanmax(np.abs(values)) if values.size else 0.0
    if max_abs <= (2.0 * np.pi + 1.0):
        return np.degrees(values)
    return values


def absolute_segment_angles(segment_pose: np.ndarray, sequence: str = "xyz") -> np.ndarray:
    # Convert the absolute segment pose into Euler angles for reporting.
    segment_rt = Rototrans(segment_pose)
    segment_rt_t = Rototrans.from_transposed_rototrans(segment_rt)
    angles = np.asarray(Angles.from_rototrans(segment_rt_t, sequence))
    return _ensure_degrees(angles)


def relative_segment_angles(parent_pose: np.ndarray, child_pose: np.ndarray, sequence: str = "xyz") -> np.ndarray:
    # Compute the child pose relative to the parent pose, then extract joint angles.
    parent_rt = Rototrans(parent_pose)
    child_rt = Rototrans(child_pose)
    parent_rt_t = Rototrans.from_transposed_rototrans(parent_rt)
    rel_rt = np.einsum("ijt, jkt -> ikt", parent_rt_t, child_rt)
    rel_rt = Rototrans(rel_rt)
    angles = np.asarray(Angles.from_rototrans(rel_rt, sequence))
    return _ensure_degrees(angles)


def build_mot_dataframe(c3d_obj: Dict) -> pd.DataFrame:
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

    # Absolute pelvis angles are used as the OpenSim pelvis columns.
    pelvis_angles = absolute_segment_angles(pelvis)
    # Lumbar motion is the torso relative to the pelvis.
    lumbar_angles = relative_segment_angles(torso, pelvis)

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
        "pelvis_list": np.round(pelvis_angles[2, 0, :], 2),
        "pelvis_rotation": np.round(pelvis_angles[0, 0, :], 2),
        "pelvis_tilt": np.round(pelvis_angles[1, 0, :], 2),
        "hip_flexion_r": np.round(hip_angles_r[0, 0, :], 2),
        "hip_adduction_r": np.round(hip_angles_r[1, 0, :], 2),
        "hip_rotation_r": np.round(hip_angles_r[2, 0, :], 2),
        "knee_angle_r": np.round(knee_angles_r[0, 0, :], 2),
        "ankle_angle_r": np.round(ankle_angles_r[0, 0, :], 2),
        "hip_flexion_l": np.round(hip_angles_l[0, 0, :], 2),
        "hip_adduction_l": np.round(hip_angles_l[1, 0, :], 2),
        "hip_rotation_l": np.round(hip_angles_l[2, 0, :], 2),
        "knee_angle_l": np.round(knee_angles_l[0, 0, :], 2),
        "ankle_angle_l": np.round(ankle_angles_l[0, 0, :], 2),
        # Convert pelvis translation from millimeters to meters for OpenSim.
        "pelvis_ty": np.round(pelvis[2, 3, :] / 1000.0, 2),
        "lumbar_bending": np.round(lumbar_angles[0, 0, :], 2),
        "lumbar_rotation": np.round(lumbar_angles[2, 0, :], 2),
        "lumbar_extension": np.round(lumbar_angles[1, 0, :], 2),
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
