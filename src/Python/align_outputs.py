"""Align an IK .mot and a GRF .mot that come from two separate C3D exports
of the same trial, so both describe the same span of real-world time before
they're used together in OpenSim Inverse Dynamics.

Standalone from run_pipeline.py: it re-derives both motion tables from their
source C3D files (so it always aligns the true underlying data, not whatever
happens to be on disk) and writes new, time-aligned .mot files alongside the
originals. It does not touch run_pipeline.py's own (unaligned) outputs.

Why this exists: two C3D exports of one trial can have very different
durations and starting points -- e.g. a markerless kinematics export
re-numbered from its own frame 1, alongside a marker+force export trimmed to
a shorter window of the same session. Assuming both files' t=0 are the same
instant, or just truncating to the shorter duration, silently applies ground
reaction forces to the wrong part of the gait cycle. Instead this reads each
source C3D's TRIAL:ACTUAL_START_FIELD / ACTUAL_END_FIELD -- the frame range
each export occupies on the original, un-cropped recording's shared frame
clock -- and intersects them to find the window the two sources actually
have in common. See workflow_utils.get_actual_frame_range /
align_ik_and_grf_dataframes for the implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opensim_scaling import write_external_loads_xml, write_inverse_dynamics_setup_xml
from workflow_utils import (
    DEFAULT_GCS_ROTATIONS,
    REQUIRED_ROTATION_LABELS,
    align_ik_and_grf_dataframes,
    build_gcs_to_opensim_rotation,
    build_grf_dataframe,
    build_mot_dataframe,
    detect_grf_plate_feet,
    find_missing_labels,
    get_actual_frame_range,
    get_force_platform_count,
    get_frame_rate_and_count,
    get_rotation_labels,
    get_segment_positions_opensim,
    load_theia_c3d,
    parse_axis_angle_rotations,
    validate_mot_dataframe,
    write_mot,
)

# USER: body names for a gait2392-style model. Change these if --scale-model
# in run_pipeline.py points at a model whose feet are named differently.
FOOT_BODY_NAMES = {"r": "calcn_r", "l": "calcn_l"}


def parse_args() -> argparse.Namespace:
    default_ik_c3d = Path(__file__).resolve().parents[2] / "sample_data" / "c3d_trials" / "JoggingL1_filt_ID.c3d"
    default_grf_c3d = Path(__file__).resolve().parents[2] / "sample_data" / "c3d_trials" / "JoggingL1_ID.c3d"
    default_ik_out = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" / "JoggingL1_filt_ID_aligned.mot"
    default_grf_out = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" / "JoggingL1_ID_grf_aligned.mot"
    default_ext_loads_out = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" / "JoggingL1_ID_external_loads.xml"
    # USER: matches run_pipeline.py's own --output-osim default -- point this
    #       at wherever your scaled model actually ended up if you changed that.
    default_scaled_model = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" / "scaled_model_ID_test.osim"
    default_id_setup_out = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" / "JoggingL1_ID_inverse_dynamics_setup.xml"

    parser = argparse.ArgumentParser(description="Align an IK and a GRF C3D export to their shared time window.")
    parser.add_argument("--ik-c3d", type=Path, default=default_ik_c3d,
                        help="Theia3D C3D file with kinematic (rotation) data.")
    parser.add_argument("--grf-c3d", type=Path, default=default_grf_c3d,
                        help="C3D file with FORCE_PLATFORM data.")
    parser.add_argument("--output-ik-mot", type=Path, default=default_ik_out,
                        help="Output path for the aligned IK .mot file.")
    parser.add_argument("--output-grf-mot", type=Path, default=default_grf_out,
                        help="Output path for the aligned GRF .mot file.")
    parser.add_argument("--gcs-rot", nargs="*", default=None, metavar="AXIS:DEG",
                        help=("Ordered axis-angle rotations transforming the source laboratory "
                              "GCS into the OpenSim GCS. Example: --gcs-rot Z:-90 X:-90. "
                              "Must match whatever run_pipeline.py used for these same sources."))
    parser.add_argument("--grf-force-threshold", type=float, default=20.0,
                        help="Vertical force (N) below which a force-platform sample is zeroed.")
    parser.add_argument("--output-external-loads", type=Path, default=default_ext_loads_out,
                        help="Output path for the generated ExternalLoads settings XML.")
    parser.add_argument("--no-external-loads", action="store_true",
                        help="Skip generating the ExternalLoads XML (aligned .mot files are still written).")
    parser.add_argument("--plate-foot", nargs="*", default=None, metavar="N:L|R",
        # USER: override when the auto-detected foot is wrong for a plate --
        #       it's a proximity heuristic (nearest foot to the plate's
        #       peak-force center of pressure), not ground truth. Also use
        #       this for a plate that's never loaded above the force
        #       threshold, which auto-detection can't assign at all.
                        help=("Force a plate's foot assignment instead of auto-detecting it, e.g. "
                              "--plate-foot 1:R 2:L. Overrides auto-detection only for the plate "
                              "numbers listed; other plates are still auto-detected."))
    parser.add_argument("--scaled-model", type=Path, default=default_scaled_model,
                        help="Scaled .osim model file (produced by run_pipeline.py) to reference "
                             "in the generated InverseDynamicsTool settings XML.")
    parser.add_argument("--output-id-setup", type=Path, default=default_id_setup_out,
                        help="Output path for the generated InverseDynamicsTool settings XML.")
    parser.add_argument("--no-id-setup", action="store_true",
                        help="Skip generating the InverseDynamicsTool settings XML.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    gcs_rotations = list(DEFAULT_GCS_ROTATIONS) if args.gcs_rot is None else parse_axis_angle_rotations(args.gcs_rot)
    gcs_to_opensim_R = build_gcs_to_opensim_rotation(gcs_rotations)

    if not args.ik_c3d.exists():
        print(f"[FAIL] IK C3D file not found: {args.ik_c3d}")
        return 1
    if not args.grf_c3d.exists():
        print(f"[FAIL] GRF C3D file not found: {args.grf_c3d}")
        return 1

    ik_c3d_obj = load_theia_c3d(args.ik_c3d)
    missing = find_missing_labels(get_rotation_labels(ik_c3d_obj), REQUIRED_ROTATION_LABELS)
    if missing:
        print("[FAIL] IK C3D is missing required rotation labels:")
        for label in missing:
            print(f"  - {label}")
        return 1

    grf_c3d_obj = load_theia_c3d(args.grf_c3d, extract_force_platforms=True)
    if get_force_platform_count(grf_c3d_obj) == 0:
        print(f"[FAIL] No FORCE_PLATFORM data in {args.grf_c3d}")
        return 1

    ik_start, ik_end = get_actual_frame_range(ik_c3d_obj)
    grf_start, grf_end = get_actual_frame_range(grf_c3d_obj)
    print(f"- IK source master frame range:  {ik_start}-{ik_end} ({args.ik_c3d.name})")
    print(f"- GRF source master frame range: {grf_start}-{grf_end} ({args.grf_c3d.name})")

    ik_df = build_mot_dataframe(ik_c3d_obj, gcs_to_opensim_R=gcs_to_opensim_R)
    grf_df = build_grf_dataframe(grf_c3d_obj, gcs_to_opensim_R=gcs_to_opensim_R, force_threshold=args.grf_force_threshold)

    ik_aligned, grf_aligned = align_ik_and_grf_dataframes(ik_df, ik_c3d_obj, grf_df, grf_c3d_obj)

    overlap_start = max(ik_start, grf_start)
    overlap_end = min(ik_end, grf_end)
    print(f"- Shared master frame window: {overlap_start}-{overlap_end} "
          f"({overlap_end - overlap_start + 1} frames)")
    print("")

    issues = validate_mot_dataframe(ik_aligned)
    if issues:
        print("[FAIL] Aligned IK data has issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    ik_out_path = write_mot(ik_aligned, args.output_ik_mot)
    print(f"[PASS] Wrote aligned IK MOT file: {ik_out_path}")
    print(f"- Rows: {ik_aligned.shape[0]}, duration: {ik_aligned['time'].iloc[-1]:.3f}s")
    print("")

    grf_out_path = write_mot(grf_aligned, args.output_grf_mot, in_degrees=False)
    print(f"[PASS] Wrote aligned GRF MOT file: {grf_out_path}")
    print(f"- Rows: {grf_aligned.shape[0]}, duration: {grf_aligned['time'].iloc[-1]:.3f}s")
    print("")

    # -------------------------------------------------------------------
    # ExternalLoads settings XML: maps each force plate to the body (foot)
    # it applied force to, which is what OpenSim's ID/IK tools actually read
    # -- the GRF .mot alone doesn't say which foot a plate belongs to.
    # -------------------------------------------------------------------
    if not args.no_external_loads:
        ik_rate, _ = get_frame_rate_and_count(ik_c3d_obj)
        ik_offset = overlap_start - ik_start
        r_foot_pos = get_segment_positions_opensim(ik_c3d_obj, "r_foot_4X4", gcs_to_opensim_R)[:, ik_offset: ik_offset + len(ik_aligned)]
        l_foot_pos = get_segment_positions_opensim(ik_c3d_obj, "l_foot_4X4", gcs_to_opensim_R)[:, ik_offset: ik_offset + len(ik_aligned)]

        assignment = detect_grf_plate_feet(grf_aligned, r_foot_pos, l_foot_pos, foot_pos_rate=ik_rate)

        overrides: dict[int, str] = {}
        if args.plate_foot:
            for spec in args.plate_foot:
                idx_str, _, foot_str = spec.partition(":")
                overrides[int(idx_str)] = foot_str.strip().lower()[0]
        assignment.update(overrides)

        plate_body_names: dict[int, str] = {}
        for plate_idx, foot in sorted(assignment.items()):
            source = "override" if plate_idx in overrides else "auto-detected"
            if foot is None:
                print(f"- Plate {plate_idx}: never loaded above the force threshold -- skipped "
                      f"(use --plate-foot {plate_idx}:L or {plate_idx}:R to force an assignment)")
                continue
            body_name = FOOT_BODY_NAMES[foot]
            plate_body_names[plate_idx] = body_name
            print(f"- Plate {plate_idx}: {foot.upper()} foot ({body_name}) [{source}]")

        if plate_body_names:
            ext_loads_path = write_external_loads_xml(args.output_external_loads, grf_out_path, plate_body_names)
            print(f"[PASS] Wrote ExternalLoads settings XML: {ext_loads_path}")
            print("- Foot assignment is a proximity heuristic -- verify it before running Inverse Dynamics.")
            print("")

            # ---------------------------------------------------------------
            # InverseDynamicsTool settings XML: ties the scaled model, the
            # aligned IK coordinates, and the ExternalLoads XML above into one
            # file ready to open in the OpenSim GUI or run directly. time_range
            # is the aligned IK's own span -- always the tighter of the two
            # (see align_ik_and_grf_dataframes), so it's guaranteed to fall
            # inside the GRF data's range too.
            # ---------------------------------------------------------------
            if not args.no_id_setup:
                if not args.scaled_model.exists():
                    print(f"- Note: --scaled-model does not exist yet ({args.scaled_model}) -- "
                          "writing the setup XML anyway; run run_pipeline.py's scaling step first.")
                id_setup_path = write_inverse_dynamics_setup_xml(
                    output_path=args.output_id_setup,
                    model_file=args.scaled_model,
                    coordinates_file=ik_out_path,
                    external_loads_file=ext_loads_path,
                    time_range=(0.0, float(ik_aligned["time"].iloc[-1])),
                    results_dir=args.output_id_setup.parent,
                )
                print(f"[PASS] Wrote InverseDynamicsTool settings XML: {id_setup_path}")
        else:
            print("[SKIP] ExternalLoads XML: no plate could be assigned a foot.")
            print("[SKIP] InverseDynamicsTool settings XML: requires ExternalLoads XML above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
