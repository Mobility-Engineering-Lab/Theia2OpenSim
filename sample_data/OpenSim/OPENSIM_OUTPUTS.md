# OpenSim Outputs Guide

This folder stores OpenSim files produced by the Python conversion workflow.

## Current outputs

- `OS_from_script.mot`: dynamic motion file exported by `run_pipeline.py`.
- `static_from_script.trc`: static virtual-marker file for OpenSim scaling.

## How files are generated

Generate both files in one run:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Lwalking7.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --static-c3d sample_data/Lwalking7.c3d \
  --static-frames 290:310 \
  --output-trc sample_data/OpenSim/static_from_script.trc
```

Related model/scaling assets live in `sample_data/` (for example `gait2392_simbody.osim` and `Scaling_Setup.xml`).
