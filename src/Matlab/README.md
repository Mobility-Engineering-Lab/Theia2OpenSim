# Theia3D → OpenSim — MATLAB Workflow

MATLAB port of the Theia2OpenSim pipeline. Reads Theia3D-exported C3D files,
computes joint kinematics, and writes OpenSim-ready `.mot` and static `.trc` files.

---

## Requirements

- **MATLAB R2016b or later** (uses `strtrim`, `fieldnames`, basic GUI dialogs)
- **BTK (Biomechanical ToolKit) for MATLAB**
  1. Download the latest release from <https://github.com/Biomechanical-ToolKit/BTKCore/releases>
  2. Unzip and add the `BTK/bin` folder to your MATLAB path:
     ```matlab
     addpath('C:/path/to/BTK/bin')
     savepath   % optional: makes it permanent
     ```
  3. Verify with `btkReadAcquisition` — if it runs without error, BTK is ready.

> **Note on 4×4 matrices:** Theia3D stores full 6-DOF pose matrices (4×4, including
> translation) in the C3D ROTATION block. If your version of BTK returns only 3×3
> rotation matrices (9 values/frame), joint angles will be correct but pelvis
> translations (pelvis\_tx/ty/tz) will be zero. In that case, use the
> [ezc3d MATLAB bindings](https://github.com/pyomeca/ezc3d) instead and point
> `load_theia_c3d.m` at `ezc3dRead`.

---

## Quick Start

1. Make sure this folder (`src/Matlab`) is on your MATLAB path, **or** open MATLAB
   with this folder as the current directory.
2. Run:
   ```matlab
   main()
   ```
3. Follow the four GUI dialogs (details below).

---

## Step-by-step GUI flow

### Dialog 1 — Static C3D (optional)
A file picker appears first. You have two options:

| Action | Effect |
|--------|--------|
| Select a dedicated `Static.c3d` | Segment origins are extracted from that file |
| **Cancel / close** | A frame from the dynamic trial is used instead |

### Dialog 2 — Static frame number
Only shown when no dedicated static file was selected.

Enter the **1-based** frame number where the subject is standing still.
The default is **300** (equivalent to Python frame index 299).

### Dialog 3 — Dynamic trial C3D
Select your walking/running/activity C3D file (e.g. `Walking.c3d`).

### Dialog 4 — Output options

| Field | Default | Notes |
|-------|---------|-------|
| Trim zeros (yes/no) | `no` | Trims leading/trailing zero-padded frames per signal |
| Output folder | same folder as the dynamic trial | Leave blank to use the default |

---

## Outputs

Both files are saved in the chosen output folder, named after the dynamic trial:

| File | Description |
|------|-------------|
| `<trial>_OS.mot` | OpenSim motion file — joint angles + pelvis pose, tab-delimited |
| `<trial>_static.trc` | OpenSim static calibration file — virtual marker positions repeated 6×  |

A 16-panel figure is also displayed showing all computed joint kinematics.

---

## File reference

| File | Purpose |
|------|---------|
| `main.m` | Entry point — GUI dialogs, calls all steps |
| `load_theia_c3d.m` | Reads C3D via BTK; returns a `c3d_data` struct |
| `get_segment_pose.m` | Extracts a named segment's `[4×4×N]` pose stack |
| `euler_xyz.m` | XYZ Cardan angle extraction (equivalent to pyomeca) |
| `ensure_degrees.m` | Converts radians → degrees when needed |
| `absolute_segment_angles.m` | Pelvis global angles (`R^T` convention) |
| `relative_segment_angles.m` | Joint angles (`R_parent^T · R_child`) |
| `build_mot_table.m` | Assembles the full kinematics struct |
| `write_mot.m` | Writes the `.mot` file with OpenSim header |
| `write_static_trc.m` | Writes the static `.trc` file |
| `trim_zeros.m` | Trims zero-padded edges and rebases time to 0 |

---

## Coordinate conventions

Theia3D and OpenSim use different axis orientations. The conversion applied:

| Theia3D axis (row index) | OpenSim axis | Kinematic use |
|--------------------------|--------------|---------------|
| Row 1 (0-based) | X | Forward / pelvis\_tx |
| Row 2 (0-based) | Y | Vertical / pelvis\_ty |
| Row 0 (0-based) | Z | Lateral / pelvis\_tz |

Translations are converted from **mm → m** during export.

---

## Calling functions directly

You can bypass the GUI and call individual steps from the command line:

```matlab
% Load files
dyn  = load_theia_c3d('path/to/Walking.c3d');
stat = load_theia_c3d('path/to/Static.c3d');   % optional

% Compute kinematics
mot = build_mot_table(dyn);

% (Optional) trim zero edges
mot = trim_zeros(mot, dyn.frame_rate);

% Save outputs
write_mot(mot, 'output/Walking_OS.mot');
write_static_trc(stat, 1, 'output/Walking_static.trc');
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `btkReadAcquisition` not found | BTK not on MATLAB path | `addpath('BTK/bin')` |
| `No ROTATION.LABELS found` | File is not a Theia3D export | Check C3D source |
| Pelvis tx/ty/tz are all zero | BTK returning 3×3 only | Switch to ezc3d bindings |
| `Segment "x_4X4" not found` | Label mismatch | Print `c3d_data.labels` to check available names |
| Figure does not appear | No display / headless | Call `build_mot_table` + `write_mot` directly |
