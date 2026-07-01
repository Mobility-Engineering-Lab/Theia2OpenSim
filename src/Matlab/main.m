function main()
% MAIN  Theia3D → OpenSim conversion pipeline with GUI file selection.
%
% USAGE:
%   main()    from the MATLAB command window or by pressing Run in the editor.
%
% REQUIRES:
%   BTK (Biomechanical ToolKit) for MATLAB on the MATLAB path.
%   Download: https://github.com/Biomechanical-ToolKit/BTKCore/releases
%
% WORKFLOW (mirrors the Jupyter notebook):
%   1. (Optional) Select a dedicated static C3D file.
%      Cancel the dialog to use a frame from the dynamic trial instead.
%   2. Enter the static frame number (1-based; default 300).
%      Only prompted when no static file was selected.
%   3. Select the dynamic trial C3D file.
%   4. Choose output options (trim zeros, output folder).
%   5. Joint angles are computed, plotted, and saved as:
%        <trial_name>_OS.mot     — OpenSim motion file
%        <trial_name>_static.trc — OpenSim static calibration file

% -------------------------------------------------------------------------
% Step 0: Verify BTK is available
% -------------------------------------------------------------------------
if ~exist('btkReadAcquisition', 'file')
    error(['BTK (Biomechanical ToolKit) is not on the MATLAB path.\n' ...
           'Download it from:\n' ...
           '  https://github.com/Biomechanical-ToolKit/BTKCore/releases\n' ...
           'Then add the BTK/bin folder to your MATLAB path.']);
end

% -------------------------------------------------------------------------
% Step 1: Optional static C3D file
% -------------------------------------------------------------------------
[sf, sp] = uigetfile('*.c3d', ...
    '(Optional) Select static C3D file — Cancel to use a dynamic trial frame');

has_static_file = ischar(sf) && ~isequal(sf, 0);

if has_static_file
    static_c3d_path = fullfile(sp, sf);
    fprintf('Loading static C3D: %s\n', static_c3d_path);
    static_c3d = load_theia_c3d(static_c3d_path);
    select_frame = [];   % not needed when dedicated static file is available
else
    static_c3d   = [];
    select_frame = [];
    fprintf('No static file selected — a frame from the dynamic trial will be used.\n');
end

% -------------------------------------------------------------------------
% Step 2: Static frame number (only when no dedicated static file)
% -------------------------------------------------------------------------
if ~has_static_file
    answer = inputdlg( ...
        {'Static frame number (1-based; default 300):'}, ...
        'Static Frame', 1, {'300'});
    if isempty(answer)
        fprintf('Cancelled.\n');
        return;
    end
    select_frame = round(str2double(strtrim(answer{1})));
    if isnan(select_frame) || select_frame < 1
        error('Invalid frame number: %s', answer{1});
    end
end

% -------------------------------------------------------------------------
% Step 3: Dynamic trial C3D file (required)
% -------------------------------------------------------------------------
[df, dp] = uigetfile('*.c3d', 'Select dynamic trial C3D file');
if isequal(df, 0)
    fprintf('No dynamic trial selected. Exiting.\n');
    return;
end
dyn_c3d_path = fullfile(dp, df);
fprintf('Loading dynamic trial: %s\n', dyn_c3d_path);
dyn_c3d = load_theia_c3d(dyn_c3d_path);

% -------------------------------------------------------------------------
% Step 4: Output options
% -------------------------------------------------------------------------
opts = inputdlg( ...
    {'Trim leading/trailing zeros? (yes / no)', ...
     'Output folder (leave blank to save next to the dynamic trial):'}, ...
    'Output Options', 1, {'no', ''});

if isempty(opts)
    fprintf('Cancelled.\n');
    return;
end

trim_flag  = strcmpi(strtrim(opts{1}), 'yes');
out_folder = strtrim(opts{2});
if isempty(out_folder)
    out_folder = dp;
end
if ~exist(out_folder, 'dir')
    mkdir(out_folder);
end

[~, trial_name] = fileparts(df);
mot_path = fullfile(out_folder, [trial_name '_OS.mot']);
trc_path = fullfile(out_folder, [trial_name '_static.trc']);

% -------------------------------------------------------------------------
% Step 5: Compute joint angles
% -------------------------------------------------------------------------
fprintf('Computing joint kinematics ...\n');
mot = build_mot_table(dyn_c3d);

if trim_flag
    fprintf('Trimming zero edges ...\n');
    mot = trim_zeros(mot, dyn_c3d.frame_rate);
end

% -------------------------------------------------------------------------
% Step 6: Plot results
% -------------------------------------------------------------------------
plot_kinematics(mot);

% -------------------------------------------------------------------------
% Step 7: Save .mot file
% -------------------------------------------------------------------------
write_mot(mot, mot_path);
fprintf('[PASS] MOT file saved: %s\n', mot_path);

% -------------------------------------------------------------------------
% Step 8: Save static .trc file
% -------------------------------------------------------------------------
if has_static_file
    src_c3d = static_c3d;
    % Use frame 1 of the static trial (the whole file is a static pose)
    static_frame = 1;
else
    src_c3d      = dyn_c3d;
    static_frame = select_frame;
end

write_static_trc(src_c3d, static_frame, trc_path);
fprintf('[PASS] Static TRC saved: %s\n', trc_path);

fprintf('\nDone.\n');
end


% =========================================================================
%  Local helper: plotting
% =========================================================================
function plot_kinematics(t)
% PLOT_KINEMATICS  16-panel figure of all computed joint angles.

time = t.time;

PANELS = { ...
    'Pelvis Tilt',              t.pelvis_tilt;        ...
    'Pelvis List',              t.pelvis_list;        ...
    'Pelvis Rotation',          t.pelvis_rotation;    ...
    'Lumbar Flex/Ext',          t.lumbar_extension;   ...
    'Lumbar Bending',           t.lumbar_bending;     ...
    'Lumbar Rotation',          t.lumbar_rotation;    ...
    'L Hip Flex/Ext',           t.hip_flexion_l;      ...
    'L Hip Add/Abd',            t.hip_adduction_l;    ...
    'L Hip Int/Ext Rot',        t.hip_rotation_l;     ...
    'R Hip Flex/Ext',           t.hip_flexion_r;      ...
    'R Hip Add/Abd',            t.hip_adduction_r;    ...
    'R Hip Int/Ext Rot',        t.hip_rotation_r;     ...
    'L Knee Flex/Ext',          t.knee_angle_l;       ...
    'R Knee Flex/Ext',          t.knee_angle_r;       ...
    'L Ankle Dorsi/Plantar',    t.ankle_angle_l;      ...
    'R Ankle Dorsi/Plantar',    t.ankle_angle_r       };

n     = size(PANELS, 1);
ncols = 4;
nrows = ceil(n / ncols);

figure('Name', 'Theia2OpenSim — Joint Kinematics', ...
       'NumberTitle', 'off', ...
       'Position', [40, 40, 1440, 860]);

for i = 1:n
    subplot(nrows, ncols, i);
    plot(time, PANELS{i, 2}, 'LineWidth', 1.2);
    title(PANELS{i, 1}, 'FontSize', 9);
    xlabel('Time (s)',   'FontSize', 8);
    ylabel('Angle (deg)','FontSize', 8);
    grid on;
    box off;
end

sgtitle('Theia3D \rightarrow OpenSim Joint Kinematics', 'FontSize', 12);
end
