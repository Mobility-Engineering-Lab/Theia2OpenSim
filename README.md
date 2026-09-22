# Theia2OpenSim

Theia2OpenSim is a Python workflow for converting Theia3D C3D exports into OpenSim motion files, with optional static TRC generation and OpenSim model scaling.

## What It Does

- Loads Theia3D C3D files
- Verifies required segment rotation labels
- Rotates the laboratory coordinate system into the OpenSim ground frame (`--gcs-rot`)
- Computes pelvis, hip, knee, ankle, and lumbar kinematics
- Exports OpenSim-compatible motion files (`.mot`)
- Optionally generates a static calibration TRC (`.trc`) from a dedicated static trial or selected frames
- Optionally runs OpenSim's Scale Tool (via the OpenSim Python API) to produce a scaled model (`.osim`)

## Repository Layout

```text
Theia2OpenSim/
├── CITATION.cff
├── LICENSE
├── README.md
├── environment.yml
├── notebook/
│   ├── Angle Calculator.ipynb
│   ├── joint_centers.mat
│   ├── OS.mot
│   └── static1.trc
├── sample_data/
│   ├── gait2392_simbody.osim
│   ├── markerstheia.xml
│   ├── Scaling_Setup.xml
│   ├── c3d_trials/
│   │   ├── LJogging.c3d
│   │   ├── LSLDJ.c3d
│   │   ├── LWalking1_filt.c3d
│   │   ├── RJogging.c3d
│   │   ├── RSLDJ.c3d
│   │   ├── RWalking1_filt.c3d
│   │   └── Static.c3d
│   ├── Orientation_test/
│   │   ├── Hopping_BL_filt.c3d
│   │   ├── Hopping_BR_filt.c3d
│   │   ├── Hopping_FL_filt.c3d
│   │   ├── Squat_BL_filt.c3d
│   │   ├── Squat_BR_filt.c3d
│   │   └── Squat_FL_filt.c3d
│   └── OpenSim_output/
└── src/
    ├── Matlab/
    │   └── placeholder_workflow.m
    └── Python/
        ├── opensim_scaling.py
        ├── run_pipeline.py
        ├── validate_workflow.py
        └── workflow_utils.py
```

## Environment Setup

Model scaling runs through the OpenSim Python API, not a separate `opensim-cmd` CLI.
`opensim` is a regular dependency in `environment.yml` (installed from conda-forge,
alongside `ezc3d` from the same channel -- see the comment in `environment.yml` for why
that pairing matters), so creating the Conda environment is the only setup step; there is
nothing else to install or add to `PATH`.

```bash
conda env create -f environment.yml -n Theia2OpenSim
conda activate Theia2OpenSim
```

If the environment already exists and you're picking up a change to `environment.yml`,
recreate it rather than trying to update in place -- `opensim` needs Python 3.11+, so an
existing 3.10 environment can't be upgraded onto it:

```bash
conda env remove -n Theia2OpenSim
conda env create -f environment.yml -n Theia2OpenSim
```

Check that it works:

```bash
conda activate Theia2OpenSim
python -c "import opensim; print(opensim.GetVersion())"
```

### Dependencies

The Python scripts in `src/Python` directly depend on:

- `numpy`
- `pandas`
- `ezc3d`
- `pyomeca`
- `opensim` (the OpenSim Python API, used for model scaling)

All of these are installed by `environment.yml`. See that file's trailing comment for why
`ezc3d` is installed from conda-forge rather than pip -- the pip wheel and OpenSim's own
package crash on import together.

## Script Entry Points

- `src/Python/run_pipeline.py`
  End-to-end pipeline: dynamic `.mot`, optional static `.trc`, optional scaling to `.osim`, in one command.
- `src/Python/validate_workflow.py`
  Validates required labels and generated motion-table integrity.
- `src/Python/workflow_utils.py`
  Shared parsing, kinematics, validation, and export utilities used by both scripts above (not run directly).
- `src/Python/opensim_scaling.py`
  Builds and runs an OpenSim ScaleTool through the OpenSim Python API, and exports a setup XML afterward as a human-readable artifact (not run directly).

## Validate Workflow

`validate_workflow.py` defaults to:

- `sample_data/c3d_trials/LWalking1_filt.c3d`

Run with default input:

```bash
python src/Python/validate_workflow.py
```

Or pass a specific trial explicitly:

```bash
python src/Python/validate_workflow.py --c3d sample_data/c3d_trials/LWalking1_filt.c3d
```

Expected output:

- Input summary (file, frame rate, frame count)
- Required-label check
- PASS/FAIL validation status

## Run The Pipeline

`--subject-mass` has no default and is always required -- it becomes the scaled model's
total mass, and downstream inverse dynamics depends on it being the actual subject's mass,
not a placeholder. Run with script defaults otherwise:

```bash
python src/Python/run_pipeline.py --subject-mass 70.0
```

Current default paths in `run_pipeline.py`:

- `--c3d`: `sample_data/Orientation_test/Hopping_BR_filt.c3d`
- `--output-mot`: `sample_data/OpenSim_output/Orientation_test_output/Hopping_BR_filt_rot_test_correct.mot`
- `--static-c3d`: `sample_data/c3d_trials/Static.c3d`
- `--output-trc`: `sample_data/OpenSim_output/Orientation_test_output/Static_rot_test_correct.trc`
- `--scale-model`: `sample_data/gait2392_simbody.osim`
- `--marker-set`: `sample_data/markerstheia.xml`
- `--output-osim`: `sample_data/OpenSim_output/Orientation_test_output/scaled_model_rot_test_correct.osim`
- `--gcs-rot`: `Z:-90 X:-90` (see [Source GCS -> OpenSim GCS Rotation](#source-gcs---opensim-gcs-rotation))

When static input is provided, scaling runs by default unless `--no-scale` is set.

## Common Examples

Dynamic `.mot` only:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/c3d_trials/LWalking1_filt.c3d \
  --output-mot sample_data/OpenSim_output/LWalking1_filt.mot \
  --no-static-c3d \
  --subject-mass 70.0
```

Dynamic `.mot` + static `.trc` + scaled `.osim` from dedicated static trial (this is what
the defaults-only form above also does, since `--static-c3d` defaults to
`sample_data/c3d_trials/Static.c3d`):

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/c3d_trials/LWalking1_filt.c3d \
  --output-mot sample_data/OpenSim_output/LWalking1_filt.mot \
  --static-c3d sample_data/c3d_trials/Static.c3d \
  --output-trc sample_data/OpenSim_output/Static.trc \
  --output-osim sample_data/OpenSim_output/scaled_model.osim \
  --subject-mass 70.0
```

Skip scaling, keep `.mot` + `.trc`:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/c3d_trials/LWalking1_filt.c3d \
  --static-c3d sample_data/c3d_trials/Static.c3d \
  --output-trc sample_data/OpenSim_output/Static.trc \
  --no-scale \
  --subject-mass 70.0
```

Use dynamic-trial frames as static source:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/c3d_trials/LWalking1_filt.c3d \
  --output-mot sample_data/OpenSim_output/LWalking1_filt.mot \
  --no-static-c3d \
  --static-frames 290:310 \
  --output-trc sample_data/OpenSim_output/Static.trc \
  --subject-mass 70.0
```

Trim leading/trailing zero regions:

```bash
python src/Python/run_pipeline.py --trim-zeros --subject-mass 70.0
```

## Source GCS -> OpenSim GCS Rotation

`--gcs-rot` specifies the ordered axis-angle rotations that transform the source
laboratory global coordinate system (GCS) into the OpenSim ground frame (`+X` anterior,
`+Y` up, `+Z` right), so trials recorded in a lab with a different GCS convention convert
correctly. It takes one or more `AXIS:DEG` pairs, applied in the order given (up to three):

```bash
python src/Python/run_pipeline.py --gcs-rot Z:-90 X:-90
```

Omitting the flag entirely uses the default, `Z:-90 X:-90`. Passing `--gcs-rot` with no pairs after it forces the identity rotation
(no transform), distinct from omitting the flag. Each rotation is an active, fixed-axis
rotation following the right-hand rule; the run stops with a `WorkflowValidationError` if
the resulting transform isn't a valid rotation. The same rotation is applied to the dynamic
`.mot`, the static `.trc`, and the coordinate `.mot` used for scaling, so all three stay in
one frame.

## `run_pipeline.py` Flags

| Flag                       | Default                                          | Purpose                                                                             |
| -------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `--c3d`                  | `sample_data/Orientation_test/Hopping_BR_filt.c3d` | Dynamic trial C3D to convert.                                                       |
| `--output-mot`           | `sample_data/OpenSim_output/Orientation_test_output/Hopping_BR_filt_rot_test_correct.mot` | Output dynamic`.mot` path.                                                        |
| `--trim-zeros`           | off                                              | Trim leading/trailing all-zero regions per signal and rebase time.                  |
| `--static-c3d`           | `sample_data/c3d_trials/Static.c3d`            | Dedicated static C3D used for TRC generation and scaling.                           |
| `--no-static-c3d`        | off                                              | Ignore`--static-c3d` and use dynamic-trial static frames or `.mot`-only flow.   |
| `--static-frames`        | `None` (runtime fallback uses `"300"`)       | Frame spec for static pose (`"300"`, `"290:310"`, `"290,300,310"`).           |
| `--output-trc`           | `sample_data/OpenSim_output/Orientation_test_output/Static_rot_test_correct.trc` | Output static`.trc` path.                                                         |
| `--repeat-static-frames` | `6`                                            | Number of repeated frames written to static`.trc`.                                |
| `--no-scale`             | off                                              | Skip OpenSim Scale Tool run.                                                        |
| `--scale-model`          | `sample_data/gait2392_simbody.osim`            | Generic model used as Scale Tool input.                                             |
| `--marker-set`           | `sample_data/markerstheia.xml`                 | MarkerSet XML matching virtual marker names.                                        |
| `--output-osim`          | `sample_data/OpenSim_output/Orientation_test_output/scaled_model_rot_test_correct.osim` | Output scaled`.osim` path.                                                        |
| `--subject-mass`         | *(required)*                                   | Subject mass (kg). Sets the scaled model's total mass; always required, whether or not scaling runs, since downstream inverse dynamics depends on it. |
| `--gcs-rot`              | `Z:-90 X:-90`                                  | Ordered `AXIS:DEG` rotations, source lab GCS -> OpenSim GCS (see [above](#source-gcs---opensim-gcs-rotation)). Up to three. |

## License

MIT License.

## Disclaimer

This toolbox is for research use. Validate coordinate transforms and generated OpenSim files before scientific or clinical interpretation.
