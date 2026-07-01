function angles_deg = relative_segment_angles(parent_pose, child_pose)
% RELATIVE_SEGMENT_ANGLES  Joint angles of child relative to parent segment.
%
% Mirrors Python workflow_utils.relative_segment_angles():
%   parent_t = transpose(parent)
%   rel_rt   = parent_t * child          (einsum 'ijt,jkt->ikt')
%   angles   = Angles.from_rototrans(rel_rt, 'xyz')
%
% INPUTS:
%   parent_pose  - [4 x 4 x N] pose matrices of the parent segment
%   child_pose   - [4 x 4 x N] pose matrices of the child segment
%
% OUTPUT:
%   angles_deg   - [3 x N] joint angles in DEGREES
%                   row 1 = ax  (flexion/extension axis)
%                   row 2 = ay  (adduction/abduction axis)
%                   row 3 = az  (internal/external rotation axis)

R_parent = parent_pose(1:3, 1:3, :);   % [3 x 3 x N]
R_child  = child_pose(1:3, 1:3, :);    % [3 x 3 x N]

N     = size(R_parent, 3);
R_rel = zeros(3, 3, N);

% Relative rotation: R_parent^T * R_child  (one frame at a time)
for k = 1:N
    R_rel(:, :, k) = R_parent(:, :, k)' * R_child(:, :, k);
end

angles_rad = euler_xyz(R_rel);          % [3 x N] in radians
angles_deg = ensure_degrees(angles_rad);
end
