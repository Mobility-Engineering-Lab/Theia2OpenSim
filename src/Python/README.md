# Python scripts

This folder contains script-based replacements for the notebook workflow.

## Files

- `validate_workflow.py`: checks required C3D labels and validates generated OpenSim table content.
- `run_pipeline.py`: converts Theia3D C3D data into an OpenSim `.mot` file.
- `workflow_utils.py`: shared processing and export functions.

## Quick start

```bash
python src/Python/validate_workflow.py
python src/Python/run_pipeline.py --output-mot sample_data/OpenSim/OS_from_script.mot
```

Optional trimming:

```bash
python src/Python/run_pipeline.py --trim-zeros
```
