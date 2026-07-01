function write_static_trc(c3d_data, select_frame, filepath)
% WRITE_STATIC_TRC  Write a static OpenSim .trc file from Theia3D segment origins.
%
% Mirrors Python workflow_utils.write_static_trc_from_c3d().
%
% Virtual marker positions are extracted from segment pose translations at
% the chosen static frame and then repeated 6 times — mimicking the static
% calibration trial format that OpenSim's Scale Tool expects.
%
% Axis mapping (Theia3D → OpenSim / TRC):
%   Theia row 2 (MATLAB row 2) → TRC X   (forward)
%   Theia row 3 (MATLAB row 3) → TRC Y   (vertical)
%   Theia row 1 (MATLAB row 1) → TRC Z   (lateral)
% (0-based Python axis_order = (1, 2, 0)  →  1-based MATLAB rows [2, 3, 1])
%
% INPUTS:
%   c3d_data     - struct from load_theia_c3d
%   select_frame - integer (1-based), frame where the subject stands still
%                  (Python default 300 ≈ MATLAB frame 300 for 1-based arrays)
%   filepath     - full path for the output .trc file

% Marker name → Theia segment label
MARKER_MAP = { ...
    'RASIS',  'r_thigh_4X4'; ...
    'LASIS',  'l_thigh_4X4'; ...
    'RKNEE',  'r_shank_4X4'; ...
    'LKNEE',  'l_shank_4X4'; ...
    'RANKLE', 'r_foot_4X4';  ...
    'LANKLE', 'l_foot_4X4';  ...
    'RTOE',   'r_toes_4X4';  ...
    'LTOE',   'l_toes_4X4';  ...
    'PELVIS', 'pelvis_4X4'   };

n_markers    = size(MARKER_MAP, 1);
frame_rate   = c3d_data.frame_rate;
total_frames = c3d_data.total_frames;

% Validate static frame (1-based)
if select_frame < 1 || select_frame > total_frames
    error('theia2opensim:badFrame', ...
        'select_frame (%d) is out of range [1, %d].', select_frame, total_frames);
end

% Theia-to-OpenSim row indices (1-based) for the translation column
AXIS_ORDER = [2, 3, 1];  % → [TRC_X, TRC_Y, TRC_Z]

% Extract marker positions at the static frame (positions in mm)
marker_xyz = zeros(n_markers, 3);
for m = 1:n_markers
    seg_label = MARKER_MAP{m, 2};
    pose      = get_segment_pose(c3d_data, seg_label);   % [4 x 4 x N]
    origin    = pose(1:3, 4, select_frame);               % [3 x 1] Theia coords
    marker_xyz(m, :) = origin(AXIS_ORDER)';               % apply axis remap
end

REPEAT_FRAMES = 6;
times = (0 : REPEAT_FRAMES - 1) / frame_rate;

% Create output directory if needed
parent_dir = fileparts(filepath);
if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
    mkdir(parent_dir);
end

[~, fname, fext] = fileparts(filepath);

fid = fopen(filepath, 'w');
if fid < 0
    error('theia2opensim:cannotWrite', 'Cannot open file for writing: %s', filepath);
end

% ---- TRC Header (3 lines) -----------------------------------------------
fprintf(fid, 'PathFileType\t4\t(X/Y/Z)\t%s%s\n', fname, fext);
fprintf(fid, 'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n');
fprintf(fid, '%.6f\t%.6f\t%d\t%d\tmm\t%.6f\t1\t%d\n', ...
    frame_rate, frame_rate, REPEAT_FRAMES, n_markers, frame_rate, REPEAT_FRAMES);

% ---- Marker-name header -------------------------------------------------
fprintf(fid, 'Frame#\tTime');
for m = 1:n_markers
    fprintf(fid, '\t%s\t\t', MARKER_MAP{m, 1});
end
fprintf(fid, '\n');

% ---- XYZ sub-header -----------------------------------------------------
fprintf(fid, '\t');   % blank Frame# and Time columns
for m = 1:n_markers
    fprintf(fid, '\tX%d\tY%d\tZ%d', m, m, m);
end
fprintf(fid, '\n');

% ---- Data rows (repeated static pose) -----------------------------------
for f = 1:REPEAT_FRAMES
    fprintf(fid, '%d\t%.6f', f, times(f));
    for m = 1:n_markers
        fprintf(fid, '\t%.6f\t%.6f\t%.6f', ...
            marker_xyz(m, 1), marker_xyz(m, 2), marker_xyz(m, 3));
    end
    fprintf(fid, '\n');
end

fclose(fid);
end
