"""Validate notebook-derived Theia3D to OpenSim workflow on a C3D input."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow_utils import (
    REQUIRED_ROTATION_LABELS,
    build_mot_dataframe,
    find_missing_labels,
    get_frame_rate_and_count,
    get_rotation_labels,
    load_theia_c3d,
    validate_mot_dataframe,
)


def parse_args() -> argparse.Namespace:
    # USER: replace the path below with your own C3D file,
    #       or pass it on the command line: --c3d "C:/MyData/trial01.c3d"
    default_c3d = Path(__file__).resolve().parents[2] / "sample_data" / "c3d_trials" / "LWalking1_filt.c3d"
    parser = argparse.ArgumentParser(description="Validate Theia3D-to-OpenSim workflow.")
    parser.add_argument("--c3d", type=Path, default=default_c3d, help="Path to Theia3D C3D file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.c3d.exists():
        print(f"[FAIL] C3D file not found: {args.c3d}")
        return 1

    # Load the sample file and confirm the required Theia rotation labels exist.
    c3d_obj = load_theia_c3d(args.c3d)
    labels = get_rotation_labels(c3d_obj)
    missing = find_missing_labels(labels, REQUIRED_ROTATION_LABELS)
    frame_rate, total_frames = get_frame_rate_and_count(c3d_obj)

    # Print a compact summary so validation is easy to scan from the terminal.
    print("Validation summary")
    print(f"- Input file: {args.c3d}")
    print(f"- Frame rate: {frame_rate:.3f} Hz")
    print(f"- Total frames: {total_frames}")
    print(f"- Rotation labels found: {len(labels)}")

    if missing:
        print("[FAIL] Missing required rotation labels:")
        for label in missing:
            print(f"  - {label}")
        return 1

    # Build the motion table and check it for basic integrity issues.
    df = build_mot_dataframe(c3d_obj)
    issues = validate_mot_dataframe(df)

    if issues:
        print("[FAIL] Data validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("[PASS] Workflow validation completed successfully.")
    print(f"- MOT table shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
