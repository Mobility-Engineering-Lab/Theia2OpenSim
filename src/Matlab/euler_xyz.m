function angles = euler_xyz(R)
% EULER_XYZ  Extract XYZ Cardan (intrinsic) Euler angles from a stack of
%            rotation matrices.
%
% Convention:  R = Rx(ax) * Ry(ay) * Rz(az)   [intrinsic XYZ = extrinsic ZYX]
% This matches pyomeca Angles.from_rototrans(rt, 'xyz').
%
% INPUT:
%   R       - [3 x 3 x N] or [4 x 4 x N] rotation/pose matrix stack.
%             A single [3 x 3] or [4 x 4] matrix is also accepted.
%
% OUTPUT:
%   angles  - [3 x N] matrix of Euler angles in RADIANS:
%               row 1 = ax  (rotation about x)
%               row 2 = ay  (rotation about y)
%               row 3 = az  (rotation about z)
%
% Extraction formulas (derived from expanding R = Rx * Ry * Rz):
%   ay =  asin( R(1,3))
%   ax =  atan2(-R(2,3), R(3,3))
%   az =  atan2(-R(1,2), R(1,1))
%
% Gimbal-lock guard fires when |cos(ay)| < 1e-6 (ay ~ ±90 deg).

% Strip homogeneous row/column if a 4x4 matrix was passed
if size(R, 1) == 4
    R = R(1:3, 1:3, :);
end

% Accept a single 3x3 matrix (size returns [3 3] with no 3rd dim)
if ndims(R) == 2
    R = reshape(R, 3, 3, 1);
end

N      = size(R, 3);
angles = zeros(3, N);

for k = 1:N
    Rk = R(:, :, k);

    % y-axis angle
    sin_ay = max(-1.0, min(1.0, Rk(1, 3)));   % clamp for numerical safety
    ay     = asin(sin_ay);

    if abs(cos(ay)) > 1e-6
        % Standard (non-degenerate) case
        ax = atan2(-Rk(2, 3),  Rk(3, 3));
        az = atan2(-Rk(1, 2),  Rk(1, 1));
    else
        % Gimbal lock: ay ≈ ±π/2 — set az = 0 and solve for ax
        az = 0;
        if ay > 0
            ax =  atan2( Rk(2, 1),  Rk(2, 2));
        else
            ax = -atan2( Rk(2, 1),  Rk(2, 2));
        end
    end

    angles(:, k) = [ax; ay; az];
end
end
