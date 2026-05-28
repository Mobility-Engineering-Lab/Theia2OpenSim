# theia2opensim

**theia2opensim** converts Theia3D markerless motion-capture outputs into OpenSim-compatible files for biomechanical analysis.

## Status

This repository started as a notebook-based workflow and now includes script-based Python tools under `src/Python` for reproducible validation and export.

## Current capabilities

- Read Theia3D-exported C3D files
- Extract segment pose matrices and sanitize missing values
- Compute pelvis, hip, knee, ankle, and lumbar kinematics
- Build OpenSim-compatible `.mot` tables
- Validate the workflow against required Theia rotation labels

## Repository structure (current)

```text
theia2opensim/
├── README.md
├── LICENSE
├── environment.yml
├── notebook/
│   └── Angle Calculator.ipynb
├── sample_data/
│   ├── Lwalking7.c3d
│   ├── markerstheia.xml
│   └── OpenSim/
│       └── README.md
└── src/
    ├── Python/
    │   ├── README.md
    │   ├── workflow_utils.py
    │   ├── validate_workflow.py
    │   └── run_pipeline.py
    └── Matlab/
        └── placeholder_workflow.m
```

## Environment setup

A conda environment file is included at `environment.yml`.

Create the environment:

```bash
conda env create -f environment.yml
conda activate theia2opensim
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

Generate an OpenSim `.mot` file from the sample C3D:

```bash
python src/Python/run_pipeline.py --output-mot sample_data/OpenSim/OS_from_script.mot
```

Optional trimming of leading/trailing zeros:

```bash
python src/Python/run_pipeline.py --trim-zeros
```

## OpenSim files

- Marker definition file: `sample_data/markerstheia.xml`
- Script output folder: `sample_data/OpenSim/`

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
