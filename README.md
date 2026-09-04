# Theia2OpenSim

Theia2OpenSim is a Python workflow for converting Theia3D C3D exports into OpenSim motion files, with optional static TRC generation and OpenSim model scaling.

## What It Does

- Loads Theia3D C3D files
- Verifies required segment rotation labels
- Computes pelvis, hip, knee, ankle, and lumbar kinematics
- Exports OpenSim-compatible motion files (`.mot`)
- Optionally generates a static calibration TRC (`.trc`) from a dedicated static trial or selected frames
- Optionally runs OpenSim Scale Tool (`opensim-cmd run-tool`) to produce a scaled model (`.osim`)

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
│   └── OpenSim_output/
│       ├── LJogging_XZ.mot
│       ├── LSLDJ_XZ.mot
│       ├── LWalking1_XZ.mot
│       ├── RJogging_XZ.mot
│       ├── RSLDJ_XZ.mot
│       ├── RWalking1_XZ.mot
│       ├── scaled_model.osim
│       ├── scaled_model_Scaling_Setup.xml
│       ├── scaled_model_Scaling_Setup_static_ik.mot
│       ├── scaled_model_static_coords.mot
│       └── Static.trc
└── src/
    ├── Matlab/
    │   └── placeholder_workflow.m
    └── Python/
        ├── opensim_scaling.py
        ├── PYTHON_TOOLBOX.md
        ├── run_pipeline.py
        ├── validate_workflow.py
        └── workflow_utils.py
```

## Environment Setup

```bash
conda env create -f environment.yml -n Theia2OpenSim
conda activate Theia2OpenSim
```

## Script Entry Points

- `src/Python/run_pipeline.py`
  - End-to-end pipeline: dynamic `.mot`, optional static `.trc`, optional scaling to `.osim`.
- `src/Python/validate_workflow.py`
  - Validates required labels and generated motion-table integrity.

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

Run with script defaults:

```bash
python src/Python/run_pipeline.py
```

Current default paths in `run_pipeline.py`:

- `--c3d`: `sample_data/c3d_trials/RJogging.c3d`
- `--output-mot`: `sample_data/OpenSim_output/RJogging_test.mot`
- `--static-c3d`: `sample_data/c3d_trials/Static.c3d`
- `--output-trc`: `sample_data/OpenSim_output/Static.trc`
- `--scale-model`: `sample_data/gait2392_simbody.osim`
- `--marker-set`: `sample_data/markerstheia.xml`
- `--output-osim`: `sample_data/OpenSim_output/scaled_model.osim`

When static input is provided, scaling runs by default unless `--no-scale` is set.
If `opensim-cmd` is not found, scaling is skipped and `.mot`/`.trc` export still completes.

## Common Examples

Dynamic `.mot` only:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/c3d_trials/LWalking1_filt.c3d \
  --output-mot sample_data/OpenSim_output/LWalking1_filt.mot \
  --no-static-c3d
```

Dynamic `.mot` + static `.trc` + scaled `.osim` from dedicated static trial:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/c3d_trials/LWalking1_filt.c3d \
  --output-mot sample_data/OpenSim_output/LWalking1_filt.mot \
  --static-c3d sample_data/c3d_trials/Static.c3d \
  --output-trc sample_data/OpenSim_output/Static.trc \
  --output-osim sample_data/OpenSim_output/scaled_model.osim
```

Skip scaling, keep `.mot` + `.trc`:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/c3d_trials/LWalking1_filt.c3d \
  --static-c3d sample_data/c3d_trials/Static.c3d \
  --output-trc sample_data/OpenSim_output/Static.trc \
  --no-scale
```

Use dynamic-trial frames as static source:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/c3d_trials/LWalking1_filt.c3d \
  --output-mot sample_data/OpenSim_output/LWalking1_filt.mot \
  --no-static-c3d \
  --static-frames 290:310 \
  --output-trc sample_data/OpenSim_output/Static.trc
```

Use an explicit OpenSim CLI path:

```bash
python src/Python/run_pipeline.py --opensim-cmd "C:/OpenSim 4.5/bin/opensim-cmd.exe"
```

## `run_pipeline.py` Flags

| Flag                       | Default                                          | Purpose                                                                             |
| -------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `--c3d`                  | `sample_data/c3d_trials/RJogging.c3d`          | Dynamic trial C3D to convert.                                                       |
| `--output-mot`           | `sample_data/OpenSim_output/RJogging_test.mot` | Output dynamic`.mot` path.                                                        |
| `--trim-zeros`           | off                                              | Trim leading/trailing all-zero regions per signal and rebase time.                  |
| `--static-c3d`           | `sample_data/c3d_trials/Static.c3d`            | Dedicated static C3D used for TRC generation and scaling.                           |
| `--no-static-c3d`        | off                                              | Ignore`--static-c3d` and use dynamic-trial static frames or `.mot`-only flow.   |
| `--static-frames`        | `None` (runtime fallback uses `"300"`)       | Frame spec for static pose (`"300"`, `"290:310"`, `"290,300,310"`).           |
| `--output-trc`           | `sample_data/OpenSim_output/Static.trc`        | Output static`.trc` path.                                                         |
| `--repeat-static-frames` | `6`                                            | Number of repeated frames written to static`.trc`.                                |
| `--no-scale`             | off                                              | Skip OpenSim Scale Tool run.                                                        |
| `--scale-model`          | `sample_data/gait2392_simbody.osim`            | Generic model used as Scale Tool input.                                             |
| `--marker-set`           | `sample_data/markerstheia.xml`                 | MarkerSet XML matching virtual marker names.                                        |
| `--output-osim`          | `sample_data/OpenSim_output/scaled_model.osim` | Output scaled`.osim` path.                                                        |
| `--subject-mass`         | `75.1646`                                      | Subject mass (kg) written into Scale Tool setup XML.                                |
| `--opensim-cmd`          | `None`                                         | Path to`opensim-cmd(.exe)`; fallback order: flag, `OPENSIM_CMD`, then `PATH`. |

## Dependency Notes

The Python scripts in `src/Python` directly depend on:

- `numpy`
- `pandas`
- `ezc3d`
- `pyomeca`

`opensim-cmd` is an external OpenSim CLI executable and is not installed by `environment.yml`.

## License

MIT License.

## Disclaimer

This toolbox is for research use. Validate coordinate transforms and generated OpenSim files before scientific or clinical interpretation.
