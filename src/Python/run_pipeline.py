"""Run a script-based Theia3D to OpenSim conversion and export a .mot file."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from opensim_ik import parse_coordinate_ranges_deg, run_ik_tool, unwrap_ik_motion, write_ik_setup_xml
from opensim_scaling import run_scale_tool, write_scale_setup_xml
from workflow_utils import (
    DEFAULT_TRC_MARKER_SEGMENT_MAP,
    REQUIRED_ROTATION_LABELS,
    build_mot_dataframe,
    find_missing_labels,
    get_frame_rate_and_count,
    get_rotation_labels,
    get_segment_reference_pose,
    load_theia_c3d,
    parse_frame_indices,
    read_mot,
    trim_dataframe,
    validate_mot_dataframe,
    write_dynamic_trc_from_c3d,
    write_mot,
    write_static_trc_from_c3d,
)


def parse_args() -> argparse.Namespace:
    # -------------------------------------------------------------------------
    # USER: edit the three lines below to point to your own files,
    #       OR leave them as-is and pass paths on the command line instead
    #       (e.g. --c3d "C:/MyData/trial01.c3d").
    # -------------------------------------------------------------------------
    default_c3d         = Path(__file__).resolve().parents[2] / "sample_data" / "Walking.c3d"
    default_out         = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim" / "OS_from_script.mot"
    default_static_c3d  = Path(__file__).resolve().parents[2] / "sample_data" / "Static.c3d"
    default_static_trc  = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim" / "Static.trc"
    default_scale_model = Path(__file__).resolve().parents[2] / "sample_data" / "gait2392_simbody.osim"
    default_marker_set  = Path(__file__).resolve().parents[2] / "sample_data" / "markerstheia.xml"
    default_output_osim = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim" / "scaled_model.osim"

    parser = argparse.ArgumentParser(description="Convert Theia3D C3D data into an OpenSim MOT file.")
    parser.add_argument("--c3d", type=Path, default=default_c3d,
                        help="Path to Theia3D dynamic C3D file.")
    parser.add_argument("--output-mot", type=Path, default=default_out,
                        help="Output .mot file path.")
    parser.add_argument("--trim-zeros", action="store_true",
                        help="Trim leading and trailing zeros per signal and rebase time.")
    parser.add_argument(
        "--static-c3d",
        type=Path,
        default=default_static_c3d,
        # USER: set to your dedicated static C3D file, e.g. "sample_data/Static.c3d".
        #       Pass --no-static-c3d if you don't have one and want --static-frames
        #       (taken from the dynamic trial) instead.
        help=(
            "Path to a dedicated static C3D file for TRC generation and scaling. "
            f"Defaults to {default_static_c3d}."
        ),
    )
    parser.add_argument(
        "--no-static-c3d",
        action="store_true",
        help="Ignore --static-c3d's default; use --static-frames against the dynamic trial "
             "instead, or produce .mot only if --static-frames is also omitted.",
    )
    parser.add_argument(
        "--static-frames",
        type=str,
        default=None,
        # USER: choose a frame number (or range) where the subject is standing still,
        #       e.g. "300", "290:310", or "290,300,310".
        help=(
            'Frames to use for the static TRC (e.g. "300", "290:310", "290,300,310"). '
            'Triggers TRC generation. Defaults to "300" when --static-c3d is provided alone.'
        ),
    )
    parser.add_argument("--output-trc", type=Path, default=default_static_trc,
                        help="Output static .trc file path.")
    parser.add_argument("--repeat-static-frames", type=int, default=6,
                        help="Number of repeated frames written to the static .trc file.")
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip running OpenSim's Scale Tool. By default, the Scale Tool runs "
                             "automatically (via opensim-cmd) whenever a static source is given "
                             "(--static-c3d or --static-frames); pass this to opt out.")
    parser.add_argument("--scale-model", type=Path, default=default_scale_model,
                        help="Generic (unscaled) .osim model file to scale.")
    parser.add_argument("--marker-set", type=Path, default=default_marker_set,
                        help="OpenSim MarkerSet .xml file matching DEFAULT_TRC_MARKER_SEGMENT_MAP's marker names.")
    parser.add_argument("--output-osim", type=Path, default=default_output_osim,
                        help="Output scaled .osim model file path.")
    parser.add_argument("--subject-mass", type=float, default=75.1646,
                        help="Subject mass in kg, written into the ScaleTool setup (informational + mass distribution).")
    parser.add_argument(
        "--opensim-cmd",
        type=Path,
        default=None,
        # USER: point this at OpenSim's bin/opensim-cmd(.exe), or set the
        #       OPENSIM_CMD environment variable instead.
        help="Path to opensim-cmd(.exe). Falls back to $OPENSIM_CMD, then PATH.",
    )
    parser.add_argument(
        "--ik",
        action="store_true",
        help="Also run OpenSim's InverseKinematicsTool against a full-trial marker .trc "
             "(marker-only IK, independent of the analytic .mot). Additive: writes separate "
             "*_markers.trc / *_ik.mot files alongside the usual output. Requires a scaled "
             "model to have been produced this run (see --no-scale).",
    )
    parser.add_argument("--ik-marker-trc", type=Path, default=None,
                        help="Output path for the full-trial marker .trc used to drive IK. "
                             "Defaults to <output-mot stem>_markers.trc.")
    parser.add_argument("--ik-output-mot", type=Path, default=None,
                        help="Output path for the IK-derived .mot. Defaults to <output-mot stem>_ik.mot.")
    parser.add_argument("--ik-accuracy", type=float, default=1e-5,
                        help="IK solver accuracy passed to OpenSim's InverseKinematicsTool.")
    parser.add_argument("--ik-marker-weight", type=float, default=10.0,
                        help="IKMarkerTask weight. Should stay well above --ik-coordinate-weight "
                             "so markers, not the analytic coordinate priors, dominate the solve.")
    parser.add_argument("--ik-coordinate-weight", type=float, default=1.0,
                        help="IKCoordinateTask weight for the analytic-motion priors added to keep "
                             "IK near a physically sane pose. Set to 0 (or use --ik-no-coordinate-priors) "
                             "for pure marker-only IK.")
    parser.add_argument("--ik-no-coordinate-priors", action="store_true",
                        help="Run marker-only IK with no analytic-motion coordinate priors at all "
                             "(the original, more underdetermined behavior).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_static_c3d:
        args.static_c3d = None
    if args.ik_marker_trc is None:
        args.ik_marker_trc = args.output_mot.with_name(args.output_mot.stem + "_markers.trc")
    if args.ik_output_mot is None:
        args.ik_output_mot = args.output_mot.with_name(args.output_mot.stem + "_ik.mot")

    # -------------------------------------------------------------------
    # Step 1: Resolve static configuration first (mirrors the notebook,
    # where the static-source cell now runs before the dynamic trial is
    # loaded). If a dedicated --static-c3d was given, validate and load it
    # now; otherwise the dynamic trial (loaded in Step 2) will be used as
    # the fallback static source.
    # -------------------------------------------------------------------
    want_static = (args.static_c3d is not None) or (args.static_frames is not None)
    static_c3d_obj = None
    static_source_label = None

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
        static_source_label = str(args.static_c3d)

    # USER: "300" is the fallback frame when --static-frames is not given.
    #       Change it to a frame where your subject is standing still.
    static_frames_spec = args.static_frames if args.static_frames is not None else "300"

    # -------------------------------------------------------------------
    # Step 2: Load the dynamic trial and confirm required segment labels exist.
    # -------------------------------------------------------------------
    if not args.c3d.exists():
        print(f"[FAIL] C3D file not found: {args.c3d}")
        return 1

    c3d_obj = load_theia_c3d(args.c3d)
    labels = get_rotation_labels(c3d_obj)
    missing = find_missing_labels(labels, REQUIRED_ROTATION_LABELS)
    if missing:
        print("[FAIL] Missing required rotation labels:")
        for label in missing:
            print(f"  - {label}")
        return 1

    frame_rate, _ = get_frame_rate_and_count(c3d_obj)

    # -------------------------------------------------------------------
    # Step 3: Resolve the static source, falling back to the dynamic trial
    # when no dedicated static C3D was provided. Used both for the optional
    # TRC export below and for the pelvis reference pose.
    # -------------------------------------------------------------------
    if static_c3d_obj is None:
        static_c3d_obj = c3d_obj
        static_source_label = f"{args.c3d} (dynamic trial)"

    # -------------------------------------------------------------------
    # Step 3a: Resolve a pelvis reference pose. Pelvis tilt/list/rotation
    # are computed relative to this pose instead of Theia's raw absolute
    # rotation -- a pelvis orientation genuinely ~180 degrees from Theia's
    # identity frame has no Euler representation that fits gait2392's
    # declared +/-90 degree range for pelvis_tilt/list, so it must be
    # re-referenced, not just reformatted. Uses --static-c3d / --static-frames
    # when given; otherwise falls back to the middle frame of the dynamic
    # trial (always in range, verified to remove the wrap and the out-of-range
    # values across the test trials). Computed before the static TRC below so
    # the TRC's marker positions and the coordinate .mot used for scaling can
    # share the same reference frame -- see write_static_trc_from_c3d.
    #
    # This reference pose is used ONLY for the static TRC + scaling's
    # coordinate .mot below (Step 3/3b) -- both need to share one reference so
    # MarkerPlacer's IK isn't asked to reconcile two different poses (see
    # extract_virtual_marker_positions). It must NOT also be used for the
    # dynamic motion output: static_c3d_obj is a genuinely static calibration
    # pose (needed for accurate marker placement -- using a moving/dynamic
    # frame range here visibly mis-places movable markers like RTOE/LTOE, as
    # seen using --no-static-c3d --static-frames on a walking/jumping trial),
    # but its heading can be ~180 degrees off a given dynamic trial's own
    # heading (see project_theia2opensim_pipeline_state.md memory). The
    # dynamic motion's own reference (dynamic_reference_pose, Step 3c below)
    # is always self-derived from that trial's own middle frame instead, so
    # good marker placement and correct dynamic-motion heading don't have to
    # trade off against each other.
    # -------------------------------------------------------------------
    if args.static_c3d is not None or args.static_frames is not None:
        reference_frame_indices = parse_frame_indices(static_frames_spec)
    else:
        _, dynamic_total_frames = get_frame_rate_and_count(c3d_obj)
        reference_frame_indices = [dynamic_total_frames // 2]

    pelvis_reference_pose = get_segment_reference_pose(
        static_c3d_obj, "pelvis_4X4", reference_frame_indices
    )
    print(f"- Pelvis reference source (scaling): {static_source_label}")
    print(f"- Pelvis reference frame(s) (scaling): {[int(idx) for idx in reference_frame_indices]}")
    print("")

    if want_static:
        frame_indices = parse_frame_indices(static_frames_spec)

        trc_path = write_static_trc_from_c3d(
            c3d_obj=static_c3d_obj,
            output_path=args.output_trc,
            frame_indices=frame_indices,
            repeat_frames=args.repeat_static_frames,
            reference_pose=pelvis_reference_pose,
        )

        print(f"[PASS] Wrote OpenSim static TRC file: {trc_path}")
        print(f"- Static source: {static_source_label}")
        print(f"- Static frames used: {static_frames_spec}")
        print(f"- Repeated TRC frames: {args.repeat_static_frames}")
        print("")

    # -------------------------------------------------------------------
    # Step 3b: Run OpenSim's Scale Tool whenever a static source was given
    # (opt out with --no-scale). Needs a coordinate .mot describing the same
    # static pose as the TRC above -- built here from static_c3d_obj using
    # the same pelvis_reference_pose, so both inputs to MarkerPlacer's IK
    # describe one consistent pose (see write_static_trc_from_c3d /
    # extract_virtual_marker_positions). A missing opensim-cmd is a warning,
    # not a hard failure, since the .mot export below doesn't depend on it.
    # -------------------------------------------------------------------
    scaling_succeeded = False
    if want_static and not args.no_scale:
        opensim_cmd = args.opensim_cmd or os.environ.get("OPENSIM_CMD")
        resolved_opensim_cmd = str(opensim_cmd) if opensim_cmd is not None else shutil.which("opensim-cmd")

        if not resolved_opensim_cmd or not Path(resolved_opensim_cmd).exists():
            print("[SKIP] Scale Tool: opensim-cmd not found. Pass --opensim-cmd, set OPENSIM_CMD, "
                  "or add --no-scale to silence this.")
            print("")
        else:
            static_frame_rate, _ = get_frame_rate_and_count(static_c3d_obj)
            static_df = build_mot_dataframe(static_c3d_obj, pelvis_reference_pose=pelvis_reference_pose)
            static_coords_path = args.output_osim.with_name(args.output_osim.stem + "_static_coords.mot")
            write_mot(static_df, static_coords_path)

            setup_xml_path = args.output_osim.with_name(args.output_osim.stem + "_Scaling_Setup.xml")
            time_range = (0.0, (args.repeat_static_frames - 1) / static_frame_rate)
            write_scale_setup_xml(
                output_xml_path=setup_xml_path,
                model_file=args.scale_model,
                marker_set_file=args.marker_set,
                marker_file=args.output_trc,
                coordinate_file=static_coords_path,
                output_model_file=args.output_osim,
                time_range=time_range,
                mass=args.subject_mass,
            )

            result = run_scale_tool(setup_xml_path, args.output_osim, opensim_cmd=resolved_opensim_cmd)
            scaling_succeeded = result.success

            if result.success:
                print(f"[PASS] Wrote scaled OpenSim model: {result.output_model_file}")
                if result.marker_rms is not None:
                    print(f"- Marker error: RMS = {result.marker_rms:.4f} m, max = {result.marker_max:.4f} m ({result.marker_max_name})")
            else:
                print("[FAIL] OpenSim Scale Tool did not complete successfully:")
                print(result.stdout)
            print("")

    # -------------------------------------------------------------------
    # Step 3c: Build the analytic dynamic motion table now (moved ahead of
    # its original Step 4 spot) so it's available below as an IKCoordinateTask
    # prior for --ik. Untrimmed here regardless of --trim-zeros -- it has to
    # share the same time base as the full-trial marker .trc written for IK,
    # which is also untrimmed; trimming (if requested) happens to a copy of
    # this same df right before the final write in Step 4.
    #
    # Uses its own reference pose (dynamic_reference_pose), NOT the scaling
    # pelvis_reference_pose above -- see the Step 3a comment for why: this
    # trial's own middle frame always shares this trial's own heading, so it
    # can't hit the ~180 degree pelvis mismatch that using a separate static
    # trial's heading can.
    # -------------------------------------------------------------------
    _, dynamic_total_frames = get_frame_rate_and_count(c3d_obj)
    dynamic_reference_pose = get_segment_reference_pose(
        c3d_obj, "pelvis_4X4", [dynamic_total_frames // 2]
    )
    print(f"- Pelvis reference source (dynamic output): {args.c3d} (this trial's own middle frame)")
    print("")

    df = build_mot_dataframe(c3d_obj, pelvis_reference_pose=dynamic_reference_pose)

    # -------------------------------------------------------------------
    # Step 3d: Optionally run OpenSim's InverseKinematicsTool against a
    # full-trial marker .trc, as an independent cross-check of the analytic
    # .mot from Step 3c/4 -- deriving joint angles from marker geometry + the
    # scaled model's own joint constraints instead of our per-segment Euler
    # decomposition. Marker-only IK turned out to be underdetermined with
    # markerstheia.xml's one-point-per-segment set (see opensim_ik.py module
    # docstring / project_marker_only_ik_attempt.md memory), so by default
    # this also adds the analytic df above as low-weight IKCoordinateTask
    # priors (--ik-marker-weight >> --ik-coordinate-weight) to keep the
    # solver near a physically sane pose; --ik-no-coordinate-priors restores
    # the original pure marker-only behavior. Requires the scaled model from
    # Step 3b above.
    # -------------------------------------------------------------------
    if args.ik:
        if not scaling_succeeded:
            print("[SKIP] --ik requires a scaled model; scaling did not run above "
                  "(check --static-c3d/--static-frames, --no-scale, and opensim-cmd resolution).")
            print("")
        else:
            ik_opensim_cmd = args.opensim_cmd or os.environ.get("OPENSIM_CMD")
            resolved_ik_opensim_cmd = str(ik_opensim_cmd) if ik_opensim_cmd is not None else shutil.which("opensim-cmd")

            if not resolved_ik_opensim_cmd or not Path(resolved_ik_opensim_cmd).exists():
                print("[SKIP] InverseKinematicsTool: opensim-cmd not found. Pass --opensim-cmd or set OPENSIM_CMD.")
                print("")
            else:
                trc_path = write_dynamic_trc_from_c3d(
                    c3d_obj=c3d_obj,
                    output_path=args.ik_marker_trc,
                    reference_pose=dynamic_reference_pose,
                )
                print(f"[PASS] Wrote full-trial marker TRC for IK: {trc_path}")

                ik_time_range = (0.0, (dynamic_total_frames - 1) / frame_rate)
                subject_name = args.ik_output_mot.stem
                ik_setup_xml_path = args.ik_output_mot.with_name(subject_name + "_IK_Setup.xml")

                coordinate_file = None
                coordinate_names = None
                if not args.ik_no_coordinate_priors:
                    coordinate_file = ik_setup_xml_path.with_name(subject_name + "_coord_prior.mot")
                    write_mot(df, coordinate_file)
                    coordinate_names = [c for c in df.columns if c != "time"]
                    print(f"- Wrote analytic-motion coordinate priors for IK: {coordinate_file}")

                write_ik_setup_xml(
                    output_xml_path=ik_setup_xml_path,
                    model_file=args.output_osim,
                    marker_file=trc_path,
                    output_motion_file=args.ik_output_mot,
                    time_range=ik_time_range,
                    marker_names=list(DEFAULT_TRC_MARKER_SEGMENT_MAP),
                    accuracy=args.ik_accuracy,
                    subject_name=subject_name,
                    marker_weight=args.ik_marker_weight,
                    coordinate_file=coordinate_file,
                    coordinate_names=coordinate_names,
                    coordinate_weight=args.ik_coordinate_weight,
                )

                ik_result = run_ik_tool(
                    ik_setup_xml_path,
                    args.ik_output_mot,
                    subject_name=subject_name,
                    opensim_cmd=resolved_ik_opensim_cmd,
                )

                if ik_result.success:
                    print(f"[PASS] Wrote IK-derived OpenSim MOT file: {ik_result.output_motion_file}")
                    if ik_result.marker_error_rms_mean is not None:
                        print(f"- Marker error across trial: mean RMS = {ik_result.marker_error_rms_mean:.4f} m, "
                              f"worst-frame RMS = {ik_result.marker_error_rms_max:.4f} m")

                    # gait2392's rotational coordinates are all unclamped, so IK can
                    # report a physically-correct pose on the wrong +/-360 degree
                    # branch (confirmed empirically -- see opensim_ik.unwrap_ik_motion).
                    # Write a second, additive copy with that branch resolved.
                    ik_df = read_mot(ik_result.output_motion_file)
                    coordinate_ranges = parse_coordinate_ranges_deg(args.output_osim)
                    unwrapped_df = unwrap_ik_motion(ik_df, coordinate_ranges)
                    unwrapped_path = ik_result.output_motion_file.with_name(
                        ik_result.output_motion_file.stem + "_unwrapped.mot"
                    )
                    write_mot(unwrapped_df, unwrapped_path)
                    print(f"- Wrote +/-360 degree unwrapped IK motion: {unwrapped_path}")
                else:
                    print("[FAIL] OpenSim InverseKinematicsTool did not complete successfully:")
                    print(ik_result.stdout)
                print("")

    # -------------------------------------------------------------------
    # Step 4: Optionally trim zero-only edges off the analytic motion table
    # from Step 3c above, then validate and export.
    # -------------------------------------------------------------------
    if args.trim_zeros:
        df = trim_dataframe(df, frame_rate=frame_rate)

    issues = validate_mot_dataframe(df)
    if issues:
        print("[FAIL] Cannot export MOT due to data issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    out_path = write_mot(df, args.output_mot)
    print(f"[PASS] Wrote OpenSim MOT file: {out_path}")
    print(f"- Rows: {df.shape[0]}")
    print(f"- Columns: {df.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
