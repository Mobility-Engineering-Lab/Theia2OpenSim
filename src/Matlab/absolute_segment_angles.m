function angles_deg = absolute_segment_angles(segment_pose)
% ABSOLUTE_SEGMENT_ANGLES  XYZ Euler angles of a segment in the global frame.
%
% Mirrors Python workflow_utils.absolute_segment_angles():
%   rt_t = transpose(rt)
%   angles = Angles.from_rototrans(rt_t, 'xyz')
%
% The rotation is transposed (inverted) before decomposition, matching the
% pyomeca convention used in the workflow utility.
%
% INPUT:
%   segment_pose  - [4 x 4 x N] pose matrices from get_segment_pose
%
% OUTPUT:
%   angles_deg    - [3 x N] Euler angles in DEGREES
%                   row 1 = ax  (index [0,0,:] in Python)
%                   row 2 = ay  (index [1,0,:] in Python)
%                   row 3 = az  (index [2,0,:] in Python)

R = segment_pose(1:3, 1:3, :);   % [3 x 3 x N]
N = size(R, 3);

% Transpose each rotation matrix (= inverse for orthonormal R)
R_t = zeros(3, 3, N);
for k = 1:N
    R_t(:, :, k) = R(:, :, k)';
end

angles_rad = euler_xyz(R_t);          % [3 x N] in radians
angles_deg = ensure_degrees(angles_rad);
end
