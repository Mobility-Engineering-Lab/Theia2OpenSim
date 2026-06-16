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
    parse_frame_indices,
    trim_dataframe,
    validate_mot_dataframe,
    write_mot,
    write_static_trc_from_c3d,
)


def parse_args() -> argparse.Namespace:
    # Default paths point to the bundled sample data so the script works out of the box.
    default_c3d = Path(__file__).resolve().parents[2] / "sample_data" / "Lwalking7.c3d"
    default_out = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim" / "OS_from_script.mot"
    default_static_trc = Path(__file__).resolve().parents[2]/ "sample_data"/ "OpenSim"/ "static_from_script.trc"

    parser = argparse.ArgumentParser(description="Convert Theia3D C3D data into an OpenSim MOT file.")
    parser.add_argument("--static-c3d",type=Path,default=None,help="Optional path to Theia3D static C3D file for static .trc generation.",)
    parser.add_argument("--output-trc",type=Path,default=default_static_trc,help="Output static .trc file path.",)
    parser.add_argument("--static-frames",type=str,default="300",help='Selected static frame(s), e.g., "300", "290:310", or "290,300,310".',)
    parser.add_argument("--repeat-static-frames",type=int,default=6,help="Number of repeated frames written to the static .trc file.",)
    parser.add_argument("--c3d", type=Path, default=default_c3d, help="Path to Theia3D C3D file.")
    parser.add_argument("--output-mot", type=Path, default=default_out, help="Output .mot file path.")
    parser.add_argument("--trim-zeros", action="store_true",help="Trim leading and trailing zeros per signal and rebase time.",)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
        # Optionally write a static TRC file for OpenSim model scaling.
    if args.static_c3d is not None:
        if not args.static_c3d.exists():
            print(f"[FAIL] Static C3D file not found: {args.static_c3d}")
            return 1

        static_c3d_obj = load_theia_c3d(args.static_c3d)
        static_labels = get_rotation_labels(static_c3d_obj)
        static_missing = find_missing_labels(static_labels, REQUIRED_ROTATION_LABELS)

        if static_missing:
            print("[FAIL] Static C3D is missing required rotation labels:")
            for label in static_missing:
                print(f"  - {label}")
            return 1

        frame_indices = parse_frame_indices(args.static_frames)

        trc_path = write_static_trc_from_c3d(
            c3d_obj=static_c3d_obj,
            output_path=args.output_trc,
            frame_indices=frame_indices,
            repeat_frames=args.repeat_static_frames,
        )

        print(f"[PASS] Wrote OpenSim static TRC file: {trc_path}")
        print(f"- Static frames used: {args.static_frames}")
        print(f"- Repeated TRC frames: {args.repeat_static_frames}")
        print("")  # Add an empty line for better readability between static TRC and MOT processing sections.


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
