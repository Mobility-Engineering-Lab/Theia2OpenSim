function t_out = trim_zeros(t_in, frame_rate)
% TRIM_ZEROS  Trim leading and trailing zero-valued edges from all signals.
%
% Mirrors Python workflow_utils.trim_dataframe():
%   Each signal is trimmed independently to its first and last non-zero
%   sample.  All signals are then shortened to the shortest valid span and
%   time is rebased to start at 0.
%
% INPUTS:
%   t_in        - struct from build_mot_table
%   frame_rate  - capture rate in Hz (from c3d_data.frame_rate)
%
% OUTPUT:
%   t_out  - trimmed struct with the same fields as t_in

col_names = fieldnames(t_in);
col_names = col_names(~strcmp(col_names, 'time'));   % exclude 'time'

ref_len  = Inf;
trimmed  = struct();

for c = 1:numel(col_names)
    sig = t_in.(col_names{c})(:)';   % ensure row vector
    nz  = find(sig ~= 0);
    if isempty(nz)
        continue;   % skip all-zero channels
    end
    trimmed_sig = sig(nz(1) : nz(end));
    trimmed.(col_names{c}) = trimmed_sig;
    ref_len = min(ref_len, numel(trimmed_sig));
end

if isinf(ref_len) || ref_len == 0
    error('theia2opensim:allZeros', ...
        'All signals are zero or empty after trimming. Check your C3D data.');
end

% Rebase time starting from 0
ref_time = (0 : ref_len - 1) / frame_rate;

t_out.time = ref_time;
for c = 1:numel(col_names)
    if isfield(trimmed, col_names{c})
        t_out.(col_names{c}) = trimmed.(col_names{c})(1 : ref_len);
    end
end
end
