# Theia2OpenSim

**Theia2OpenSim** is a Python toolbox for scaling OpenSim models and driving them with Theia3D markerless motion-capture outputs.

## Status

This repository started as a notebook-based workflow and now includes script-based Python tools under `src/Python` for reproducible validation, model scaling support, and motion export.

## Current capabilities

- Read Theia3D-exported C3D files
- Extract segment pose matrices and sanitize missing values
- Compute pelvis, hip, knee, ankle, and lumbar kinematics
- Build OpenSim-compatible `.mot` tables with full pelvis translation (`pelvis_tx`, `pelvis_ty`, `pelvis_tz`)
- Generate static OpenSim-compatible `.trc` files for model scaling
- Run OpenSim's Scale Tool automatically (via `opensim-cmd`) to produce a scaled `.osim` model
- Validate the workflow against required Theia rotation labels

## Repository structure (current)

```text
Theia2OpenSim/
├── README.md
├── LICENSE
├── environment.yml
├── notebook/
│   ├── Angle Calculator.ipynb
│   ├── OS.mot
│   ├── joint_centers.mat
│   └── static1.trc
├── sample_data/
│   ├── gait2392_simbody.osim
│   ├── gait2392_simbody_scaled.osim
│   ├── Walking.c3d
│   ├── Static.c3d
│   ├── markerstheia.xml
│   ├── Scaling_Setup.xml
│   └── OpenSim/
│       ├── OPENSIM_OUTPUTS.md
│       ├── OS_from_script.mot
│       ├── Static.trc
│       ├── scaled_model.osim
│       ├── scaled_model_Scaling_Setup.xml
│       └── scaled_model_static_coords.mot
└── src/
    ├── Python/
    │   ├── PYTHON_TOOLBOX.md
    │   ├── workflow_utils.py
    │   ├── opensim_scaling.py
    │   ├── validate_workflow.py
    │   └── run_pipeline.py
    └── Matlab/
        └── placeholder_workflow.m
```

## Environment setup

A conda environment file is included at `environment.yml`.

Create the environment (the `-n` override names it `Theia2OpenSim`; omit it to use the lowercase `theia2opensim` name defined in the file):

```bash
conda env create -f environment.yml -n Theia2OpenSim
conda activate Theia2OpenSim
```

## Workflow validation

Validate the script workflow on sample data:

```bash
python src/Python/validate_workflow.py
```

Expected behavior:

- confirms required rotation labels are present
- checks generated motion table integrity
- prints PASS/FAIL summary

## Run conversion pipeline

Running with no arguments does the most common thing end to end: it converts
`sample_data/Walking.c3d` to a dynamic `.mot`, generates a static `.trc` from
`sample_data/Static.c3d` (that's the default `--static-c3d`), and — if
`opensim-cmd` can be found — runs OpenSim's Scale Tool to produce a scaled
`.osim` model:

```bash
python src/Python/run_pipeline.py
```

This writes `sample_data/OpenSim/OS_from_script.mot`, `sample_data/OpenSim/Static.trc`,
and `sample_data/OpenSim/scaled_model.osim` (plus the intermediate
`scaled_model_Scaling_Setup.xml` / `scaled_model_static_coords.mot` files the
Scale Tool needs).

`opensim-cmd` resolution order: `--opensim-cmd`, then the `OPENSIM_CMD`
environment variable, then `PATH`. If none resolve, scaling is skipped with a
`[SKIP]` message and the `.mot`/`.trc` export still completes.

### Common flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--c3d` | `sample_data/Walking.c3d` | Dynamic trial C3D to convert. |
| `--output-mot` | `sample_data/OpenSim/OS_from_script.mot` | Output dynamic `.mot` path. |
| `--trim-zeros` | off | Trim leading/trailing all-zero frames per signal and rebase time. |
| `--static-c3d` | `sample_data/Static.c3d` | Dedicated static C3D used for the TRC, the pelvis reference pose, and scaling. |
| `--no-static-c3d` | off | Ignore `--static-c3d`'s default. Falls back to `--static-frames` against the dynamic trial, or produces `.mot` only if `--static-frames` is also omitted. |
| `--static-frames` | `"300"` | Frame(s) to use for the static pose instead of/in addition to `--static-c3d`, e.g. `"300"`, `"290:310"`, `"290,300,310"`. |
| `--output-trc` | `sample_data/OpenSim/Static.trc` | Output static `.trc` path. |
| `--repeat-static-frames` | `6` | Number of repeated frames written to the static `.trc`. |
| `--no-scale` | off | Skip running the OpenSim Scale Tool (the `.mot`/`.trc` export still runs). |
| `--scale-model` | `sample_data/gait2392_simbody.osim` | Generic (unscaled) `.osim` model to scale. |
| `--marker-set` | `sample_data/markerstheia.xml` | OpenSim `MarkerSet` XML matching the virtual marker names. |
| `--output-osim` | `sample_data/OpenSim/scaled_model.osim` | Output scaled `.osim` model path. |
| `--subject-mass` | `75.1646` | Subject mass (kg) written into the Scale Tool setup. |
| `--opensim-cmd` | none | Path to `opensim-cmd(.exe)`. Falls back to `$OPENSIM_CMD`, then `PATH`. |

Optional trimming of leading/trailing zeros:

```bash
python src/Python/run_pipeline.py --trim-zeros
```

Skip scaling and just export `.mot` + static `.trc`:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --output-trc sample_data/OpenSim/static_from_script.trc \
  --no-scale
```

Generate `.mot` and static `.trc` using selected frames from the dynamic trial
instead of a dedicated static C3D (scaling still runs too, using those same
frames as the static pose, unless `--no-scale` is also passed):

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --no-static-c3d \
  --static-frames 290:310 \
  --output-trc sample_data/OpenSim/static_from_script.trc
```

## OpenSim files

- Marker definition file: `sample_data/markerstheia.xml`
- Script output folder: `sample_data/OpenSim/`
- Static `.trc` virtual markers are mapped from Theia segment origins for OpenSim scaling
- Scaling is driven by a templated Scale Tool setup XML (`src/Python/opensim_scaling.py`) run via `opensim-cmd run-tool` as a subprocess
- OpenSim sample model/setup assets are provided under `sample_data/`

## MATLAB path

`src/Matlab/placeholder_workflow.m` is a placeholder for future MATLAB parity with the Python workflow.

## Collaboration

Contributions are welcome.

High-priority collaboration areas:

- robust `.trc` export from script workflow
- batch processing for multiple trials
- improved coordinate-system documentation and validation
- OpenSim verification examples and setup templates
- cross-validation against marker-based pipelines

If you want to contribute, open an issue with:

- your use case
- sample data constraints
- expected OpenSim outputs

## Citation

If you use this toolbox in research, please cite the associated manuscript/communication when available.

## License

MIT License.

## Disclaimer

This toolbox is for research use. Users must verify coordinate transforms and exported OpenSim files before scientific or clinical interpretation.

## Example commands

Dynamic `.mot` only (skip the static `.trc` and scaling):

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --no-static-c3d
```

Dynamic `.mot` plus static `.trc` plus scaled `.osim`, from the dedicated
static trial (this is also what running with no flags does, since
`--static-c3d` already defaults to `sample_data/Static.c3d`):

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --static-c3d sample_data/Static.c3d \
  --output-trc sample_data/OpenSim/static_from_script.trc \
  --output-osim sample_data/OpenSim/scaled_model.osim
```

Same, but skip the Scale Tool run (`.mot` + `.trc` only):

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --static-c3d sample_data/Static.c3d \
  --output-trc sample_data/OpenSim/static_from_script.trc \
  --no-scale
```

Dynamic `.mot` plus static `.trc` from frames within the dynamic trial
instead of a dedicated static C3D (scaling still runs too unless `--no-scale`
is also passed):

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --no-static-c3d \
  --static-frames 290:310 \
  --output-trc sample_data/OpenSim/static_from_script.trc
```

Point at a specific `opensim-cmd` instead of relying on `PATH`:

```bash
python src/Python/run_pipeline.py --opensim-cmd "/path/to/OpenSim/bin/opensim-cmd"
```
