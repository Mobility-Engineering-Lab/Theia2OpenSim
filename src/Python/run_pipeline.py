"""Run a script-based Theia3D to OpenSim conversion and export a .mot file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from opensim_scaling import TORSO_MARKER_NAMES, read_marker_names, run_scale_tool
from workflow_utils import (
    DEFAULT_TRC_MARKER_SEGMENT_MAP,
    HEAD_ROTATION_LABEL,
    REQUIRED_ROTATION_LABELS,
    DEFAULT_GCS_ROTATIONS,
    add_missing_output_extensions,
    build_gcs_to_opensim_rotation,
    build_static_coords_dataframe,
    build_grf_dataframe,
    format_frame_list,
    parse_axis_angle_rotations,
    build_mot_dataframe,
    find_missing_labels,
    get_force_platform_count,
    get_frame_rate_and_count,
    get_rotation_labels,
    get_segment_tracked_mask,
    load_theia_c3d,
    parse_frame_indices,
    trim_dataframe,
    validate_mot_dataframe,
    write_mot,
    write_static_trc_from_c3d,
)


def build_parser() -> argparse.ArgumentParser:
    # -------------------------------------------------------------------------
    # USER: edit the default paths below to point to your own files,
    #       OR leave them as-is and pass paths on the command line instead
    #       (e.g. --c3d "C:/MyData/trial01.c3d").
    # -------------------------------------------------------------------------
    default_c3d         = Path(__file__).resolve().parents[2] / "sample_data" /"c3d_trials"/"JoggingL1_filt_ID.c3d"
    default_out         = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" /"JoggingL1_filt_ID.mot"
    default_static_c3d  = Path(__file__).resolve().parents[2] / "sample_data" / "c3d_trials"/"Static_filt_ID.c3d"
    default_static_trc  = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" /"Static_ID_test.trc"
    # The *_downshift_pelvis pair: pelvis_offset moved onto Theia's pelvis centre, and the
    # marker set has the C7 marker that torso scaling needs (markerstheia.xml has HEAD instead).
    default_scale_model = Path(__file__).resolve().parents[2] / "sample_data" / "gait2392_simbody_downshift_pelvis.osim"
    default_marker_set  = Path(__file__).resolve().parents[2] / "sample_data" / "markerstheia_downshift_pelvis.xml"
    default_output_osim = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" /"scaled_model_ID_test.osim"
    # USER: this is deliberately a *different* file from default_c3d above --
    #       force-plate (analog) data should generally not go through the same
    #       low-pass filtering used for marker/rotation kinematics, so labs
    #       often export an unfiltered C3D for force alongside a filtered one
    #       for kinematics. Point this at whichever C3D actually has your
    #       FORCE_PLATFORM data.
    default_grf_c3d      = Path(__file__).resolve().parents[2] / "sample_data" / "c3d_trials"/"JoggingL1_ID.c3d"
    default_output_grf_mot = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" /"JoggingL1_ID_grf.mot"

    parser = argparse.ArgumentParser(description="Convert Theia3D C3D data into an OpenSim MOT file.")

    # Each argument group is also one box in run_pipeline_gui.py, with its
    # inputs and outputs (dest starting "output_") shown side by side.
    subject = parser.add_argument_group("Subject and coordinate system")
    # USER: measure this for your subject. There is deliberately no default --
    #       the scaled model takes this as its total mass, so a wrong value
    #       propagates straight into inverse-dynamics joint moments.
    subject.add_argument("--subject-mass", type=float, required=True,
                         help="Subject mass in kg (required). Sets the scaled model's total mass, "
                              "which downstream inverse dynamics depends on.")
    subject.add_argument("--gcs-rot", nargs="*", default=None, metavar="AXIS:DEG",
        help=("Ordered axis-angle rotations transforming the source laboratory "
              "GCS into the OpenSim GCS. " 'Example: --gcs-rot Z:-90 X:-90. '
              "Up to three rotations may be specified. Omit the flag for the default "
              "(Z:-90 X:-90); pass it with no values for no rotation."),)

    model = parser.add_argument_group("Generic model")
    model.add_argument("--scale-model", type=Path, default=default_scale_model,
                       help="Generic (unscaled) .osim model file to scale.")
    model.add_argument("--marker-set", type=Path, default=default_marker_set,
                       help="OpenSim MarkerSet .xml file matching DEFAULT_TRC_MARKER_SEGMENT_MAP's marker names.")

    static = parser.add_argument_group("Static trial and scaling")
    static.add_argument("--static-c3d", type=Path,default=default_static_c3d,
        # USER: set to your dedicated static C3D file, e.g. "sample_data/c3d_trials/Static.c3d".
        #       Pass --no-static-c3d if you don't have one and want --static-frames
        #       (taken from the dynamic trial) instead.
                         help=("Path to a dedicated static C3D file for TRC generation and scaling. "
                              f"Defaults to {default_static_c3d}." ), )
    static.add_argument("--no-static-c3d",action="store_true",
                        help="Ignore --static-c3d's default; use --static-frames against the dynamic trial "
                             "instead, or produce .mot only if --static-frames is also omitted.",)
    static.add_argument("--static-frames",type=str,default=None,
        # USER: choose a frame number (or range) where the subject is standing still,
        #       e.g. "300", "290:310", or "290,300,310".
                        help=( 'Frames to use for the static TRC (e.g. "300", "290:310", "290,300,310"). '
                               'Triggers TRC generation. Defaults to "300" when --static-c3d is provided alone.'),)
    static.add_argument("--output-trc", type=Path, default=default_static_trc,
                        help="Output static .trc file path.")
    static.add_argument("--repeat-static-frames", type=int, default=6,
                        help="Number of repeated frames written to the static .trc file.")
    static.add_argument("--no-scale", action="store_true",
                        help="Skip running OpenSim's Scale Tool. By default, the Scale Tool runs "
                             "automatically (via the OpenSim Python API) whenever a static source is "
                             "given (--static-c3d or --static-frames); pass this to opt out.")
    static.add_argument("--no-torso", action="store_true",
        # USER: pass this when torso_4X4 isn't trustworthy for this subject/trial.
        #       The torso is scaled off PELVIS-C7 (C7 = torso_4X4's origin); a lost
        #       head_4X4 alone does not need this -- the run only warns about it.
                        help="Drop the C7 marker, torso scaling, and the lumbar marker-placement "
                             "tasks, leaving the torso at its generic size. Use when torso_4X4 "
                             "isn't trustworthy for this subject/trial.")
    static.add_argument("--output-osim", type=Path, default=default_output_osim,
                        help="Output scaled .osim model file path.")

    dynamic = parser.add_argument_group("Dynamic trial (kinematics)")
    dynamic.add_argument("--c3d", type=Path, default=default_c3d,
                        help="Path to Theia3D dynamic C3D file.")
    dynamic.add_argument("--output-mot", type=Path, default=default_out,
                        help="Output .mot file path.")
    dynamic.add_argument("--trim-zeros", action="store_true",
                        help="Trim leading and trailing zeros per signal and rebase time.")

    grf = parser.add_argument_group("Inverse dynamics (ground reaction forces)")
    grf.add_argument("--grf-c3d", type=Path, default=default_grf_c3d,
                        help="Theia3D C3D file containing force-platform data, for ground-reaction-force "
                             "export. Independent of --c3d (see the USER note on default_grf_c3d above); "
                             "runs automatically whenever it resolves to a file that exists and reports "
                             "FORCE_PLATFORM data. Pass --no-grf to skip.")
    grf.add_argument("--output-grf-mot", type=Path, default=default_output_grf_mot,
                        help="Output ground-reaction-force .mot file path.")
    grf.add_argument("--no-grf", action="store_true",
                        help="Skip ground-reaction-force export even if --grf-c3d resolves.")
    grf.add_argument("--grf-force-threshold", type=float, default=20.0,
                        help="Vertical force (N) below which a force-platform sample is treated as "
                             "unloaded and zeroed (force, moment, torque, and center of pressure). "
                             "Needed because center of pressure is undefined at zero force and noisy "
                             "near it -- see build_grf_dataframe's docstring.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def warn_lost_upper_body(c3d_obj, source_label: str, frame_indices: np.ndarray | None = None) -> None:
    # Theia can lose the head -- and, with it, possibly the upper body -- while
    # still tracking the pelvis and legs. Nothing is built from head_4X4, so a
    # lost head is only a warning sign to check the torso in Theia. Frames where
    # the pelvis is lost too (subject out of view) aren't reported.
    labels = get_rotation_labels(c3d_obj)
    pelvis = get_segment_tracked_mask(c3d_obj, "pelvis_4X4")
    frames = np.arange(pelvis.size) if frame_indices is None else np.asarray(frame_indices)
    frames = frames[pelvis[frames]]
    if HEAD_ROTATION_LABEL not in labels:
        print(f"[WARN] {source_label} has no {HEAD_ROTATION_LABEL} label, so whether Theia tracked the")
        print("       head can't be checked. Check the torso / upper body in Theia.")
        return
    for segment in ("torso_4X4", HEAD_ROTATION_LABEL):
        lost = frames[~get_segment_tracked_mask(c3d_obj, segment)[frames]]
        if lost.size:
            print(f"[WARN] {source_label}: Theia lost {segment} on {lost.size} of {frames.size} frames where "
                  f"the pelvis is tracked (frames {format_frame_list(lost)}).")
            print("       Check the torso / upper body in Theia for these frames.")


def main() -> int:
    args = parse_args()
    add_missing_output_extensions(args)

    if args.gcs_rot is None:
        gcs_rotations = list(DEFAULT_GCS_ROTATIONS)
    else:
        gcs_rotations = parse_axis_angle_rotations(
            args.gcs_rot
        )

    gcs_to_opensim_R = build_gcs_to_opensim_rotation(
        gcs_rotations
    )

    if gcs_rotations:
        rotation_text = " -> ".join(
            f"{axis}:{angle:g}°"
            for axis, angle in gcs_rotations
        )
    else:
        rotation_text = "identity"

    print(
        f"- Source GCS -> OpenSim GCS: {rotation_text}"
    )

    if args.no_static_c3d:
        args.static_c3d = None

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

    if not want_static:
        print("[WARN] No static source: --no-static-c3d is set and --static-frames is empty.")
        print("       No static .trc and no scaled .osim will be written -- only the kinematics .mot"
              + ("" if args.no_grf else " (and GRF .mot)") + ".")
        print("       To scale from the dynamic trial instead, pass --static-frames with frames where")
        print("       the subject is standing still (e.g. --static-frames 10:20).")
        print("")

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
    warn_lost_upper_body(c3d_obj, f"Dynamic trial {args.c3d.name}")

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
    if want_static:
        _, static_frame_count = get_frame_rate_and_count(static_c3d_obj)
        try:
            frame_indices = parse_frame_indices(static_frames_spec)
        except ValueError:
            print(f'[FAIL] Invalid --static-frames "{static_frames_spec}" (use e.g. "300", "290:310", or "290,300,310").')
            return 1
        if frame_indices.size == 0 or frame_indices.min() < 0 or frame_indices.max() >= static_frame_count:
            print(f'[FAIL] --static-frames "{static_frames_spec}" is outside the static source '
                  f"({static_source_label}), which has frames 0-{static_frame_count - 1}.")
            return 1

        # --no-torso drops the C7 marker from the TRC to match the
        # C7-less ScaleTool config built below (opensim_scaling.build_scale_tool's
        # include_torso=False) -- both sides need to agree, or the Scale Tool
        # setup XML would reference a marker the TRC doesn't have.
        marker_segment_map = {
            name: label for name, label in DEFAULT_TRC_MARKER_SEGMENT_MAP.items()
            if not (args.no_torso and name in TORSO_MARKER_NAMES)
        }

        # Every TRC marker comes from one Theia segment; an untracked segment
        # on a chosen frame would put that marker at the lab origin.
        untracked = {}
        for segment in sorted(set(marker_segment_map.values())):
            lost = frame_indices[~get_segment_tracked_mask(static_c3d_obj, segment)[frame_indices]]
            if lost.size:
                untracked[segment] = lost
        if untracked:
            print(f'[FAIL] Theia did not track these segments on some --static-frames "{static_frames_spec}" '
                  f"of {static_source_label}:")
            for segment, lost in untracked.items():
                print(f"  - {segment}: frames {format_frame_list(lost)}")
            print("       Pick frames where the whole body is tracked"
                  + (", or pass --no-torso if only torso_4X4 is lost." if "torso_4X4" in untracked else "."))
            return 1
        warn_lost_upper_body(static_c3d_obj, f"Static frames of {static_source_label}", frame_indices)

        if not args.no_scale:
            if not args.marker_set.exists():
                print(f"[FAIL] Marker set file not found: {args.marker_set}")
                return 1
            missing_markers = sorted(set(marker_segment_map) - read_marker_names(args.marker_set))
            if missing_markers:
                print(f"[FAIL] --marker-set {args.marker_set.name} is missing markers the static TRC uses: "
                      + ", ".join(missing_markers))
                if set(missing_markers) & TORSO_MARKER_NAMES:
                    print("       C7 (torso_4X4's origin, on /bodyset/torso) scales the torso -- see "
                          "sample_data/markerstheia_downshift_pelvis.xml -- or pass --no-torso.")
                return 1

        trc_path = write_static_trc_from_c3d(
            c3d_obj=static_c3d_obj,
            output_path=args.output_trc,
            frame_indices=frame_indices,
            repeat_frames=args.repeat_static_frames,
            marker_segment_map=marker_segment_map,
            gcs_to_opensim_R=gcs_to_opensim_R,
        )

        print(f"[PASS] Wrote OpenSim static TRC file: {trc_path}")
        print(f"- Static source: {static_source_label}")
        print(f"- Static frames used: {static_frames_spec}")
        print(f"- Repeated TRC frames: {args.repeat_static_frames}")
        if args.no_torso:
            print("- C7 marker / torso scaling: disabled (--no-torso)")
        print("")

    # -------------------------------------------------------------------
    # Step 3b: Run OpenSim's Scale Tool (via the OpenSim Python API) whenever
    # a static source was given (opt out with --no-scale). Needs a
    # coordinate .mot describing the same static pose as the TRC above --
    # built here from static_c3d_obj using the same lab_to_opensim_R, so both
    # inputs to MarkerPlacer's IK describe one consistent pose (see
    # write_static_trc_from_c3d / extract_virtual_marker_positions).
    # -------------------------------------------------------------------
    scale_failed = False
    if want_static and args.no_scale:
        print("[WARN] --no-scale is set: the static .trc was written but the Scale Tool was skipped,")
        print("       so no scaled .osim will be written. Untick --no-scale to scale the model.")
        print("")

    if want_static and not args.no_scale:
        static_frame_rate, _ = get_frame_rate_and_count(static_c3d_obj)
        static_df = build_static_coords_dataframe(
            build_mot_dataframe(static_c3d_obj, gcs_to_opensim_R=gcs_to_opensim_R,),
            frame_indices=frame_indices,
            repeat_frames=args.repeat_static_frames,
            frame_rate=static_frame_rate,
        )
        static_coords_path = args.output_osim.with_name(args.output_osim.stem + "_static_coords.mot")
        write_mot(static_df, static_coords_path)

        setup_xml_path = args.output_osim.with_name(args.output_osim.stem + "_Scaling_Setup.xml")
        time_range = (0.0, (args.repeat_static_frames - 1) / static_frame_rate)
        result = run_scale_tool(
            model_file=args.scale_model,
            marker_set_file=args.marker_set,
            marker_file=args.output_trc,
            coordinate_file=static_coords_path,
            output_model_file=args.output_osim,
            time_range=time_range,
            mass=args.subject_mass,
            setup_xml_path=setup_xml_path,
            include_torso=not args.no_torso,
        )

        if result.success:
            print(f"[PASS] Wrote scaled OpenSim model: {result.output_model_file}")
            if result.marker_rms is not None:
                print(f"- Marker error: RMS = {result.marker_rms:.4f} m, max = {result.marker_max:.4f} m ({result.marker_max_name})")
        else:
            scale_failed = True
            print("[FAIL] OpenSim Scale Tool did not complete successfully:")
            print(result.log_text)
        print("")

    # -------------------------------------------------------------------
    # Step 4: Build the dynamic motion table, optionally trim zero-only
    # edges, then validate and export.
    # -------------------------------------------------------------------
    df = build_mot_dataframe(c3d_obj,gcs_to_opensim_R=gcs_to_opensim_R,)

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
    print("")

    # -------------------------------------------------------------------
    # Step 5: Optional ground-reaction-force export, for Inverse Dynamics.
    # Standalone from Step 4 above (own C3D source, own native analog-rate
    # time base -- see build_grf_dataframe) but reuses the same
    # gcs_to_opensim_R, so both outputs describe one consistent OpenSim
    # ground frame. Missing file / no force-platform data is a [SKIP], not a
    # hard failure, since GRF is an add-on to the kinematics export above.
    # -------------------------------------------------------------------
    if not args.no_grf:
        if not args.grf_c3d.exists():
            print(f"[SKIP] GRF export: file not found: {args.grf_c3d}")
        else:
            grf_c3d_obj = load_theia_c3d(args.grf_c3d, extract_force_platforms=True)
            if get_force_platform_count(grf_c3d_obj) == 0:
                print(f"[SKIP] GRF export: no FORCE_PLATFORM data in {args.grf_c3d}")
            else:
                grf_df = build_grf_dataframe(
                    grf_c3d_obj,
                    gcs_to_opensim_R=gcs_to_opensim_R,
                    force_threshold=args.grf_force_threshold,
                )
                grf_path = write_mot(grf_df, args.output_grf_mot, in_degrees=False)
                print(f"[PASS] Wrote OpenSim GRF MOT file: {grf_path}")
                print(f"- GRF source: {args.grf_c3d}")
                print(f"- Rows: {grf_df.shape[0]}")
                print(f"- Columns: {grf_df.shape[1]}")
                print(f"- Force threshold: {args.grf_force_threshold:g} N")

    if not want_static:
        print("")
        print("[WARN] No scaled model was produced (no static source) -- see the warning at the top.")
    elif args.no_scale:
        print("")
        print("[WARN] No scaled model was produced (--no-scale) -- see the warning above.")
    if scale_failed:
        # The .mot/GRF outputs above are still written, but exit non-zero so a
        # failed scale doesn't look like a successful run (e.g. in the GUI).
        print("")
        print("[FAIL] Scaling failed -- no scaled model was written. See the Scale Tool log above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
