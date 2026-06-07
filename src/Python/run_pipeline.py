"""Run a script-based Theia3D to OpenSim conversion and export a .mot file."""

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
    trim_dataframe,
    validate_mot_dataframe,
    write_mot,
)


def parse_args() -> argparse.Namespace:
    # Default paths point to the bundled sample data so the script works out of the box.
    default_c3d = Path(__file__).resolve().parents[2] / "sample_data" / "Lwalking7.c3d"
    default_out = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim" / "OS_from_script.mot"

    parser = argparse.ArgumentParser(description="Convert Theia3D C3D data into an OpenSim MOT file.")
    parser.add_argument("--c3d", type=Path, default=default_c3d, help="Path to Theia3D C3D file.")
    parser.add_argument("--output-mot", type=Path, default=default_out, help="Output .mot file path.")
    parser.add_argument(
        "--trim-zeros",
        action="store_true",
        help="Trim leading and trailing zeros per signal and rebase time.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.c3d.exists():
        print(f"[FAIL] C3D file not found: {args.c3d}")
        return 1

    # Load the C3D input and confirm the segment labels needed by the workflow.
    c3d_obj = load_theia_c3d(args.c3d)
    labels = get_rotation_labels(c3d_obj)
    missing = find_missing_labels(labels, REQUIRED_ROTATION_LABELS)
    if missing:
        # Stop early if the source file does not contain the segment set we expect.
        print("[FAIL] Missing required rotation labels:")
        for label in missing:
            print(f"  - {label}")
        return 1

    # Build the OpenSim table, optionally trim zero-only edges, then validate it.
    frame_rate, _ = get_frame_rate_and_count(c3d_obj)
    df = build_mot_dataframe(c3d_obj)

    if args.trim_zeros:
        df = trim_dataframe(df, frame_rate=frame_rate)

    issues = validate_mot_dataframe(df)
    if issues:
        print("[FAIL] Cannot export MOT due to data issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    # Write the final OpenSim motion file to the requested location.
    out_path = write_mot(df, args.output_mot)
    print(f"[PASS] Wrote OpenSim MOT file: {out_path}")
    print(f"- Rows: {df.shape[0]}")
    print(f"- Columns: {df.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
