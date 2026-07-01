function deg = ensure_degrees(values)
% ENSURE_DEGREES  Convert to degrees if values appear to be in radians.
%
% Mirrors Python workflow_utils._ensure_degrees(): if the maximum absolute
% value is within the radian range (<= 2*pi + 1) the data is assumed to be
% in radians and is converted; otherwise it is returned unchanged.

max_abs = max(abs(values(:)));

if isempty(max_abs) || isnan(max_abs) || max_abs <= (2 * pi + 1)
    deg = rad2deg(values);
else
    deg = values;
end
end
