function t = build_mot_table(c3d_data)
% BUILD_MOT_TABLE  Compute all joint kinematics and assemble the motion struct.
%
% Mirrors Python workflow_utils.build_mot_dataframe().
%
% INPUT:
%   c3d_data  - struct from load_theia_c3d
%
% OUTPUT:
%   t  - struct with one field per MOT column (all [1 x N_frames] row vectors)
%        Column order matches the Python output exactly.
%
% Axis mapping (Theia3D → OpenSim):
%   Theia row-index 1 (0-based) → OpenSim X   (forward)
%   Theia row-index 2 (0-based) → OpenSim Y   (vertical)
%   Theia row-index 0 (0-based) → OpenSim Z   (lateral)
% In MATLAB 1-based indexing this becomes pose rows 2, 3, 1.

frame_rate   = c3d_data.frame_rate;
total_frames = c3d_data.total_frames;
time         = round((0 : total_frames - 1) / frame_rate, 5);   % [1 x N]

% -------------------------------------------------------------------------
% Load segment poses  ([4 x 4 x N_frames] each)
% -------------------------------------------------------------------------
pelvis  = get_segment_pose(c3d_data, 'pelvis_4X4');
torso   = get_segment_pose(c3d_data, 'torso_4X4');

l_thigh = get_segment_pose(c3d_data, 'l_thigh_4X4');
l_shank = get_segment_pose(c3d_data, 'l_shank_4X4');
l_foot  = get_segment_pose(c3d_data, 'l_foot_4X4');

r_thigh = get_segment_pose(c3d_data, 'r_thigh_4X4');
r_shank = get_segment_pose(c3d_data, 'r_shank_4X4');
r_foot  = get_segment_pose(c3d_data, 'r_foot_4X4');

% -------------------------------------------------------------------------
% Joint angle computations
% All functions return [3 x N_frames] in degrees:
%   row 1 = ax  (x-axis Cardan angle)
%   row 2 = ay  (y-axis Cardan angle)
%   row 3 = az  (z-axis Cardan angle)
% -------------------------------------------------------------------------

% Absolute pelvis angles
pelvis_ang  = absolute_segment_angles(pelvis);   % [3 x N]

% Lumbar: torso relative to pelvis
lumbar_ang  = relative_segment_angles(torso, pelvis);

% Hip (pelvis as parent)
l_hip_ang   = relative_segment_angles(pelvis, l_thigh);
r_hip_ang   = relative_segment_angles(pelvis, r_thigh);

% Knee
l_knee_ang  = relative_segment_angles(l_thigh, l_shank);
r_knee_ang  = relative_segment_angles(r_thigh, r_shank);

% Ankle
l_ankle_ang = relative_segment_angles(l_shank, l_foot);
r_ankle_ang = relative_segment_angles(r_shank, r_foot);

% -------------------------------------------------------------------------
% Pelvis translations: mm → m, Theia-to-OpenSim axis remapping
% In the 4x4 pose matrix the translation is in column 4 (1-indexed).
% Theia row 2 (MATLAB row 2) → OpenSim X  (forward)
% Theia row 3 (MATLAB row 3) → OpenSim Y  (vertical)
% Theia row 1 (MATLAB row 1) → OpenSim Z  (lateral)
% -------------------------------------------------------------------------
pelvis_tx = round(squeeze(pelvis(2, 4, :))' / 1000, 2);  % [1 x N]
pelvis_ty = round(squeeze(pelvis(3, 4, :))' / 1000, 2);
pelvis_tz = round(squeeze(pelvis(1, 4, :))' / 1000, 2);

% -------------------------------------------------------------------------
% Assemble output struct  (column order matches Python build_mot_dataframe)
% -------------------------------------------------------------------------
t.time             = time;
t.pelvis_list      = round(pelvis_ang(3, :), 2);
t.pelvis_rotation  = round(pelvis_ang(1, :), 2);
t.pelvis_tilt      = round(pelvis_ang(2, :), 2);
t.hip_flexion_r    = round(r_hip_ang(1, :),  2);
t.hip_adduction_r  = round(r_hip_ang(2, :),  2);
t.hip_rotation_r   = round(r_hip_ang(3, :),  2);
t.knee_angle_r     = round(r_knee_ang(1, :), 2);
t.ankle_angle_r    = round(r_ankle_ang(1, :),2);
t.hip_flexion_l    = round(l_hip_ang(1, :),  2);
t.hip_adduction_l  = round(l_hip_ang(2, :),  2);
t.hip_rotation_l   = round(l_hip_ang(3, :),  2);
t.knee_angle_l     = round(l_knee_ang(1, :), 2);
t.ankle_angle_l    = round(l_ankle_ang(1, :),2);
t.pelvis_tx        = pelvis_tx;
t.pelvis_ty        = pelvis_ty;
t.pelvis_tz        = pelvis_tz;
t.lumbar_bending   = round(lumbar_ang(1, :), 2);
t.lumbar_rotation  = round(lumbar_ang(3, :), 2);
t.lumbar_extension = round(lumbar_ang(2, :), 2);
end
