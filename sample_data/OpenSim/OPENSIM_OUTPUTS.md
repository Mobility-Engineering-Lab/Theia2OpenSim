# OpenSim Outputs Guide

This folder stores OpenSim files produced by the Python conversion workflow.

## Current outputs

- `OS_from_script.mot`: dynamic motion file exported by `run_pipeline.py`. Includes pelvis orientation (`pelvis_list`, `pelvis_rotation`, `pelvis_tilt`), full pelvis translation (`pelvis_tx`, `pelvis_ty`, `pelvis_tz`), hip/knee/ankle angles, and lumbar angles.
- `static_from_script.trc`: static virtual-marker file for OpenSim scaling.

## How files are generated

Generate both files using a dedicated static trial:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --static-c3d sample_data/Static.c3d \
  --output-trc sample_data/OpenSim/static_from_script.trc
```

If you don't have a dedicated static trial, omit `--static-c3d` and select frames from the dynamic trial instead:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Walking.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --static-frames 290:310 \
  --output-trc sample_data/OpenSim/static_from_script.trc
```

Related model/scaling assets live in `sample_data/` (for example `gait2392_simbody.osim` and `Scaling_Setup.xml`).
