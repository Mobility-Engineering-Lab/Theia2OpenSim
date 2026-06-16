# Python Toolbox Guide

This folder contains the script-based workflow used to validate Theia3D C3D inputs and export OpenSim files.

## Main scripts

- `validate_workflow.py`: validates required rotation labels and motion-table integrity.
- `run_pipeline.py`: exports dynamic `.mot` and optional static `.trc` in one command.
- `workflow_utils.py`: shared parsing, kinematics, validation, and export utilities.

## Common commands

Validate sample input:

```bash
python src/Python/validate_workflow.py --c3d sample_data/Lwalking7.c3d
```

Export dynamic `.mot`:

```bash
python src/Python/run_pipeline.py --c3d sample_data/Lwalking7.c3d --output-mot sample_data/OpenSim/OS_from_script.mot
```

Export dynamic `.mot` plus static `.trc` for scaling:

```bash
python src/Python/run_pipeline.py \
  --c3d sample_data/Lwalking7.c3d \
  --output-mot sample_data/OpenSim/OS_from_script.mot \
  --static-c3d sample_data/Lwalking7.c3d \
  --static-frames 290:310 \
  --output-trc sample_data/OpenSim/static_from_script.trc
```

Optional trimming of leading/trailing zero regions:

```bash
python src/Python/run_pipeline.py --trim-zeros
```
