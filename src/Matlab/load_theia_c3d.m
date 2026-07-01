function c3d_data = load_theia_c3d(filepath)
% LOAD_THEIA_C3D  Load a Theia3D-exported C3D file using BTK.
%
% Requires the Biomechanical ToolKit (BTK) for MATLAB.
%   Download: https://github.com/Biomechanical-ToolKit/BTKCore/releases
%   Then add the BTK/bin folder to your MATLAB path.
%
% Theia3D stores each segment's full 6-DOF pose as a 4x4 transformation
% matrix in the C3D ROTATION section (16 values per frame per segment).
% BTK returns these as [N_frames x 16] per label.  If your version of BTK
% only exposes 3x3 matrices (9 values) you will see a warning and
% translation data (pelvis_tx/ty/tz) will be zero — switch to ezc3d
% MATLAB bindings in that case: https://github.com/pyomeca/ezc3d
%
% OUTPUT struct fields:
%   frame_rate    - double, capture rate in Hz
%   total_frames  - integer, total number of frames
%   labels        - {N_seg x 1} cell array of rotation segment labels
%   pose          - [4 x 4 x N_seg x N_frames] pose matrices (double)
%                   NaN-free; bottom row is [0 0 0 1]

if ~exist(filepath, 'file')
    error('theia2opensim:fileNotFound', 'C3D file not found: %s', filepath);
end

acq = btkReadAcquisition(filepath);

frame_rate   = btkGetPointFrequency(acq);
first_frame  = btkGetFirstFrame(acq);
last_frame   = btkGetLastFrame(acq);
total_frames = last_frame - first_frame + 1;

% -------------------------------------------------------------------------
% Read rotation labels from the PARAMETER metadata block
% -------------------------------------------------------------------------
md = btkGetMetaData(acq);

try
    labels_raw = md.children.ROTATION.children.LABELS.info.values;
catch ME
    btkDeleteAcquisition(acq);
    error('theia2opensim:noRotations', ...
        ['No ROTATION.LABELS found. Is this a Theia3D C3D export?\n' ...
         'Original error: %s'], ME.message);
end

% LABELS may come back as a fixed-width char array; convert to cell
if ischar(labels_raw)
    labels = strtrim(cellstr(labels_raw));
else
    labels = strtrim(labels_raw);
end
labels = labels(:);   % ensure column vector
n_seg  = numel(labels);

% -------------------------------------------------------------------------
% Read rotation data via btkGetRotations
% BTK returns a struct with one field per label; each field is
% [N_frames x M] where M = 9 (3x3) or M = 16 (4x4, full Theia pose).
% -------------------------------------------------------------------------
[rot_struct, ~] = btkGetRotations(acq);
btkDeleteAcquisition(acq);

% Find the first valid field so we can infer the data width
mat_dim = [];
for i = 1:n_seg
    fn = matlab.lang.makeValidName(labels{i});
    if isfield(rot_struct, fn)
        n_vals  = size(rot_struct.(fn), 2);
        if n_vals == 16
            mat_dim = 4;
        elseif n_vals == 9
            mat_dim = 3;
        else
            error('theia2opensim:unexpectedFormat', ...
                'Unexpected rotation width: %d values/frame (expected 9 or 16).', n_vals);
        end
        break;
    end
end

if isempty(mat_dim)
    error('theia2opensim:noRotationData', ...
        ['btkGetRotations returned no matching fields.\n' ...
         'For Theia3D 4x4 pose data, consider ezc3d MATLAB bindings:\n' ...
         '  https://github.com/pyomeca/ezc3d']);
end

if mat_dim == 3
    warning(['theia2opensim:3x3Only\n' ...
        'BTK returned 3x3 rotation matrices — translation columns will be zero.\n' ...
        'For full 4x4 Theia3D pose data use ezc3d MATLAB bindings:\n' ...
        '  https://github.com/pyomeca/ezc3d']);
end

% -------------------------------------------------------------------------
% Build [4 x 4 x N_seg x N_frames] pose array
% C3D stores matrices column-major; reshape(vals', dim, dim, []) restores
% the correct [dim x dim x N_frames] orientation for each segment.
% -------------------------------------------------------------------------
pose = zeros(4, 4, n_seg, total_frames);
pose(4, 4, :, :) = 1;   % homogeneous bottom-right element

for i = 1:n_seg
    fn = matlab.lang.makeValidName(labels{i});
    if ~isfield(rot_struct, fn)
        warning('theia2opensim:missingSegment', ...
            'Segment "%s" not found in BTK rotation output — pose left as identity.', labels{i});
        continue;
    end

    vals = rot_struct.(fn);  % [N_frames x n_vals], double

    % Reshape: vals' is [n_vals x N_frames]; MATLAB fills column-major,
    % matching the C3D column-major storage convention.
    mat = reshape(vals', mat_dim, mat_dim, []);   % [dim x dim x N_frames]

    pose(1:mat_dim, 1:mat_dim, i, :) = mat;
end

% -------------------------------------------------------------------------
% Pack output struct
% -------------------------------------------------------------------------
c3d_data.frame_rate   = double(frame_rate);
c3d_data.total_frames = total_frames;
c3d_data.labels       = labels;
c3d_data.pose         = pose;
end
