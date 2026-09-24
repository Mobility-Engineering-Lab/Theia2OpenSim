"""Run a script-based Theia3D to OpenSim conversion and export a .mot file."""

from __future__ import annotations

import argparse
from pathlib import Path

from opensim_scaling import run_scale_tool
from workflow_utils import (
    DEFAULT_TRC_MARKER_SEGMENT_MAP,
    REQUIRED_ROTATION_LABELS,
    DEFAULT_GCS_ROTATIONS,
    build_gcs_to_opensim_rotation,
    build_grf_dataframe,
    parse_axis_angle_rotations,
    build_mot_dataframe,
    find_missing_labels,
    get_force_platform_count,
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
    # -------------------------------------------------------------------------
    # USER: edit the default paths below to point to your own files,
    #       OR leave them as-is and pass paths on the command line instead
    #       (e.g. --c3d "C:/MyData/trial01.c3d").
    # -------------------------------------------------------------------------
    default_c3d         = Path(__file__).resolve().parents[2] / "sample_data" /"c3d_trials"/"JoggingL1_filt_ID.c3d"
    default_out         = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" /"JoggingL1_filt_ID.mot"
    default_static_c3d  = Path(__file__).resolve().parents[2] / "sample_data" / "c3d_trials"/"Static_filt_ID.c3d"
    default_static_trc  = Path(__file__).resolve().parents[2] / "sample_data" / "OpenSim_output" /"Static_ID_test.trc"
    default_scale_model = Path(__file__).resolve().parents[2] / "sample_data" / "gait2392_simbody.osim"
    default_marker_set  = Path(__file__).resolve().parents[2] / "sample_data" / "markerstheia.xml"
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
    parser.add_argument("--c3d", type=Path, default=default_c3d,
                        help="Path to Theia3D dynamic C3D file.")
    parser.add_argument("--output-mot", type=Path, default=default_out,
                        help="Output .mot file path.")
    parser.add_argument("--trim-zeros", action="store_true",
                        help="Trim leading and trailing zeros per signal and rebase time.")
    parser.add_argument("--static-c3d", type=Path,default=default_static_c3d,
        # USER: set to your dedicated static C3D file, e.g. "sample_data/c3d_trials/Static.c3d".
        #       Pass --no-static-c3d if you don't have one and want --static-frames
        #       (taken from the dynamic trial) instead.
                         help=("Path to a dedicated static C3D file for TRC generation and scaling. "
                              f"Defaults to {default_static_c3d}." ), )
    parser.add_argument("--no-static-c3d",action="store_true",
                        help="Ignore --static-c3d's default; use --static-frames against the dynamic trial "
                             "instead, or produce .mot only if --static-frames is also omitted.",)
    parser.add_argument("--static-frames",type=str,default=None,
        # USER: choose a frame number (or range) where the subject is standing still,
        #       e.g. "300", "290:310", or "290,300,310".
                        help=( 'Frames to use for the static TRC (e.g. "300", "290:310", "290,300,310"). '
                               'Triggers TRC generation. Defaults to "300" when --static-c3d is provided alone.'),)
    parser.add_argument("--output-trc", type=Path, default=default_static_trc,
                        help="Output static .trc file path.")
    parser.add_argument("--repeat-static-frames", type=int, default=6,
                        help="Number of repeated frames written to the static .trc file.")
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip running OpenSim's Scale Tool. By default, the Scale Tool runs "
                             "automatically (via the OpenSim Python API) whenever a static source is "
                             "given (--static-c3d or --static-frames); pass this to opt out.")
    parser.add_argument("--no-head", action="store_true",
        # USER: pass this when head_4X4 isn't trustworthy for this subject/trial
        #       (check: some Theia captures report a constant identity transform for
        #       head_4X4 instead of NaN, i.e. it looks present but was never tracked --
        #       this corrupts torso scaling and, because MarkerPlacer solves all markers
        #       jointly, distorts nearby joint placement too. Symptom: a torso scale
        #       factor far from 1.0 and/or the largest marker error reported at HEAD.)
                        help="Drop the HEAD marker and torso scaling, reproducing scaling as it "
                             "behaved before HEAD-based torso scaling was added.")
    parser.add_argument("--scale-model", type=Path, default=default_scale_model,
                        help="Generic (unscaled) .osim model file to scale.")
    parser.add_argument("--marker-set", type=Path, default=default_marker_set,
                        help="OpenSim MarkerSet .xml file matching DEFAULT_TRC_MARKER_SEGMENT_MAP's marker names.")
    parser.add_argument("--output-osim", type=Path, default=default_output_osim,
                        help="Output scaled .osim model file path.")
    # USER: measure this for your subject. There is deliberately no default --
    #       the scaled model takes this as its total mass, so a wrong value
    #       propagates straight into inverse-dynamics joint moments.
    parser.add_argument("--subject-mass", type=float, required=True,
                        help="Subject mass in kg (required). Sets the scaled model's total mass, "
                             "which downstream inverse dynamics depends on.")
    parser.add_argument("--gcs-rot", nargs="*", default=None, metavar="AXIS:DEG",
        help=("Ordered axis-angle rotations transforming the source laboratory "
              "GCS into the OpenSim GCS. " 'Example: --gcs-rot Z:-90 X:-90. '
              "Up to three rotations may be specified."),)
    parser.add_argument("--grf-c3d", type=Path, default=default_grf_c3d,
                        help="Theia3D C3D file containing force-platform data, for ground-reaction-force "
                             "export. Independent of --c3d (see the USER note on default_grf_c3d above); "
                             "runs automatically whenever it resolves to a file that exists and reports "
                             "FORCE_PLATFORM data. Pass --no-grf to skip.")
    parser.add_argument("--output-grf-mot", type=Path, default=default_output_grf_mot,
                        help="Output ground-reaction-force .mot file path.")
    parser.add_argument("--no-grf", action="store_true",
                        help="Skip ground-reaction-force export even if --grf-c3d resolves.")
    parser.add_argument("--grf-force-threshold", type=float, default=20.0,
                        help="Vertical force (N) below which a force-platform sample is treated as "
                             "unloaded and zeroed (force, moment, torque, and center of pressure). "
                             "Needed because center of pressure is undefined at zero force and noisy "
                             "near it -- see build_grf_dataframe's docstring.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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
    if want_static:
        frame_indices = parse_frame_indices(static_frames_spec)

        # --no-head drops the HEAD marker from the TRC to match the
        # HEAD-less ScaleTool config built below (opensim_scaling.build_scale_tool's
        # include_head=False) -- both sides need to agree, or the Scale Tool
        # setup XML would reference a marker the TRC doesn't have.
        marker_segment_map = (
            {name: label for name, label in DEFAULT_TRC_MARKER_SEGMENT_MAP.items() if name != "HEAD"}
            if args.no_head else None
        )

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
        if args.no_head:
            print("- HEAD marker / torso scaling: disabled (--no-head)")
        print("")

    # -------------------------------------------------------------------
    # Step 3b: Run OpenSim's Scale Tool (via the OpenSim Python API) whenever
    # a static source was given (opt out with --no-scale). Needs a
    # coordinate .mot describing the same static pose as the TRC above --
    # built here from static_c3d_obj using the same lab_to_opensim_R, so both
    # inputs to MarkerPlacer's IK describe one consistent pose (see
    # write_static_trc_from_c3d / extract_virtual_marker_positions).
    # -------------------------------------------------------------------
    if want_static and not args.no_scale:
        static_frame_rate, _ = get_frame_rate_and_count(static_c3d_obj)
        static_df = build_mot_dataframe(static_c3d_obj, gcs_to_opensim_R=gcs_to_opensim_R,)
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
            include_head=not args.no_head,
        )

        if result.success:
            print(f"[PASS] Wrote scaled OpenSim model: {result.output_model_file}")
            if result.marker_rms is not None:
                print(f"- Marker error: RMS = {result.marker_rms:.4f} m, max = {result.marker_max:.4f} m ({result.marker_max_name})")
        else:
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
