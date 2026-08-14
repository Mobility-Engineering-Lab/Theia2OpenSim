# Python Toolbox Guide

This folder contains the script-based workflow used to validate Theia3D C3D inputs and export OpenSim files.

## Main scripts

- `validate_workflow.py`: validates required rotation labels and motion-table integrity.
- `run_pipeline.py`: exports dynamic `.mot`, optional static `.trc`, and (by default) a scaled `.osim` model in one command.
- `workflow_utils.py`: shared parsing, kinematics, validation, and export utilities.
- `opensim_scaling.py`: writes an OpenSim Scale Tool setup XML and runs it via `opensim-cmd run-tool` as a subprocess.

## Common commands

Validate sample input:

```bash
python src/Python/validate_workflow.py --c3d sample_data/Walking.c3d
```

Run everything with defaults (dynamic `.mot`, static `.trc` from `sample_data/Static.c3d`,
and a scaled `.osim` via `opensim-cmd` if it can be found):

```bash
python src/Python/run_pipeline.py
```

Export dynamic `.mot` only, skipping the static `.trc` and scaling:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --no-static-c3d
```

Export dynamic `.mot` plus static `.trc` plus scaled `.osim`, using a dedicated static trial
(this is what the no-flags form above also does, since `--static-c3d` defaults to
`sample_data/Static.c3d`):

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --static-c3d sample_data/Static.c3d \
  --output-trc sample_data/OpenSim/static_from_script.trc \
  --output-osim sample_data/OpenSim/scaled_model.osim
```

Same, but skip the Scale Tool run:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --static-c3d sample_data/Static.c3d \
  --output-trc sample_data/OpenSim/static_from_script.trc \
  --no-scale
```

Export dynamic `.mot` plus static `.trc` using frames from the dynamic trial instead
(no dedicated static C3D required; scaling still runs too unless `--no-scale` is
also passed):

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --no-static-c3d \
  --static-frames 290:310 \
  --output-trc sample_data/OpenSim/static_from_script.trc
```

Optional trimming of leading/trailing zero regions:

```bash
python src/Python/run_pipeline.py --trim-zeros
```

Point at a specific `opensim-cmd` instead of relying on `PATH`:

```bash
python src/Python/run_pipeline.py --opensim-cmd "/path/to/OpenSim/bin/opensim-cmd"
```

## `run_pipeline.py` flags

| Flag                       | Default                                    | Purpose                                                                                                                                                          |
| -------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--c3d`                  | `sample_data/Walking.c3d`                | Dynamic trial C3D to convert.                                                                                                                                    |
| `--output-mot`           | `sample_data/OpenSim/OS_from_script.mot` | Output dynamic`.mot` path.                                                                                                                                     |
| `--trim-zeros`           | off                                        | Trim leading/trailing all-zero frames per signal and rebase time.                                                                                                |
| `--static-c3d`           | `sample_data/Static.c3d`                 | Dedicated static C3D used for the TRC, the pelvis reference pose, and scaling.                                                                                   |
| `--no-static-c3d`        | off                                        | Ignore`--static-c3d`'s default. Falls back to `--static-frames` against the dynamic trial, or produces `.mot` only if `--static-frames` is also omitted. |
| `--static-frames`        | `"300"`                                  | Frame(s) for the static pose, e.g.`"300"`, `"290:310"`, `"290,300,310"`.                                                                                   |
| `--output-trc`           | `sample_data/OpenSim/Static.trc`         | Output static`.trc` path.                                                                                                                                      |
| `--repeat-static-frames` | `6`                                      | Number of repeated frames written to the static`.trc`.                                                                                                         |
| `--no-scale`             | off                                        | Skip running the OpenSim Scale Tool (`.mot`/`.trc` export still runs).                                                                                       |
| `--scale-model`          | `sample_data/gait2392_simbody.osim`      | Generic (unscaled)`.osim` model to scale.                                                                                                                      |
| `--marker-set`           | `sample_data/markerstheia.xml`           | OpenSim`MarkerSet` XML matching the virtual marker names.                                                                                                      |
| `--output-osim`          | `sample_data/OpenSim/scaled_model.osim`  | Output scaled`.osim` model path.                                                                                                                               |
| `--subject-mass`         | `75.1646`                                | Subject mass (kg) written into the Scale Tool setup.                                                                                                             |
| `--opensim-cmd`          | none                                       | Path to`opensim-cmd(.exe)`. Falls back to `$OPENSIM_CMD`, then `PATH`.                                                                                     |

Scaling only runs when a static source resolves (`--static-c3d` or `--static-frames`,
and `--no-scale` is not passed) and `opensim-cmd` can be found; otherwise scaling is
skipped with a `[SKIP]` message and the `.mot`/`.trc` export still completes.
