function pose = get_segment_pose(c3d_data, segment_label)
% GET_SEGMENT_POSE  Extract one segment's 4x4 pose matrix stack.
%
% Mirrors Python workflow_utils.get_segment_pose().
%
% INPUTS:
%   c3d_data      - struct returned by load_theia_c3d
%   segment_label - string, e.g. 'pelvis_4X4'
%
% OUTPUT:
%   pose  - [4 x 4 x N_frames] double array
%           NaNs replaced with 0; bottom row forced to [0 0 0 1]

idx = find(strcmp(c3d_data.labels, segment_label), 1);
if isempty(idx)
    error('theia2opensim:missingLabel', ...
        'Segment "%s" not found in C3D rotation labels.\nAvailable: %s', ...
        segment_label, strjoin(c3d_data.labels', ', '));
end

% c3d_data.pose is [4 x 4 x N_seg x N_frames]; squeeze the segment dimension
pose = squeeze(c3d_data.pose(:, :, idx, :));   % [4 x 4 x N_frames]

if ndims(pose) == 2
    % Single-frame file: squeeze drops the frame dimension
    pose = reshape(pose, 4, 4, 1);
end

% Replace NaN with 0 and enforce valid homogeneous bottom row
pose(isnan(pose))  = 0;
pose(4, :, :)      = 0;
pose(4, 4, :)      = 1;
end
