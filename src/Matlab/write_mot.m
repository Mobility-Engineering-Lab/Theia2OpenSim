function write_mot(t, filepath)
% WRITE_MOT  Write an OpenSim .mot file from the motion data struct.
%
% Mirrors Python workflow_utils.write_mot().
% The output file has an OpenSim-compatible text header followed by a
% tab-delimited table of joint angles and pelvis translations.
%
% INPUTS:
%   t         - struct from build_mot_table (or trim_zeros)
%   filepath  - full path for the output .mot file

% Column order must match the Python output exactly
COL_NAMES = { ...
    'time', 'pelvis_list', 'pelvis_rotation', 'pelvis_tilt', ...
    'hip_flexion_r',  'hip_adduction_r', 'hip_rotation_r', ...
    'knee_angle_r',   'ankle_angle_r', ...
    'hip_flexion_l',  'hip_adduction_l', 'hip_rotation_l', ...
    'knee_angle_l',   'ankle_angle_l', ...
    'pelvis_tx',      'pelvis_ty',     'pelvis_tz', ...
    'lumbar_bending', 'lumbar_rotation', 'lumbar_extension' };

n_cols     = numel(COL_NAMES);
n_rows     = numel(t.time);
start_time = t.time(1);
end_time   = t.time(end);

% Create parent directory if needed
parent_dir = fileparts(filepath);
if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
    mkdir(parent_dir);
end

[~, fname, fext] = fileparts(filepath);

fid = fopen(filepath, 'w');
if fid < 0
    error('theia2opensim:cannotWrite', 'Cannot open file for writing: %s', filepath);
end

% OpenSim MOT header
fprintf(fid, '%s%s\n',      fname, fext);
fprintf(fid, 'version=1\n');
fprintf(fid, 'datacolumns %d\n', n_cols);
fprintf(fid, 'datarows %d\n',    n_rows);
fprintf(fid, 'range %.5f %.5f\n', start_time, end_time);
fprintf(fid, 'inDegrees=yes\n');
fprintf(fid, 'endheader\n');

% Column-name header row
fprintf(fid, '%s', COL_NAMES{1});
for j = 2:n_cols
    fprintf(fid, '\t%s', COL_NAMES{j});
end
fprintf(fid, '\n');

% Pre-fetch column data as a [n_rows x n_cols] matrix for fast writing
data = zeros(n_rows, n_cols);
for j = 1:n_cols
    data(:, j) = t.(COL_NAMES{j})(:);
end

% Write one row per time step
fmt = ['%.6f' repmat('\t%.6f', 1, n_cols - 1) '\n'];
fprintf(fid, fmt, data');   % data' is [n_cols x n_rows]; fprintf reads column-major

fclose(fid);
end
