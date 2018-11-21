function [cost c_dist, c_angle, c_cos, c_sin] = dw_metric(halos_pred, halos_true, ref, Ks)
%DW_METRIC 
%
%     cost = dw_metric(halos_pred, halos_true, ref, Ks)
%     cost = dw_metric(halos_pred, halos_true, {masses}, Ks)
%
% Inputs:
%      halos_pred Kx2xN (or Kx2, assumed same for every case)
%      halos_true Kx2xN (or Kx2, assumed same for every case)
%             ref 2xN   (or 2x1, assumed same for every case)
% OR ref is a cell array containing KxN (or Kx1) masses
%              Ks 1xN   (default K, the maximum, for every case)
%
% Outputs:
%           cost  1x1 

% Iain Murray, December 2012
% Hackily ported from provided python code.

Kmax = size(halos_pred, 1);
assert(size(halos_true, 1) == Kmax);
N = max([size(halos_pred,3), size(halos_true,3), size(ref,2)]);

% Copy inputs N times where necessary:
if isequal(size(halos_pred), [Kmax,2])
    halos_pred = repmat(halos_pred, [1, 1, N]);
end
if isequal(size(halos_true), [Kmax,2])
    halos_true = repmat(halos_true, [1, 1, N]);
end
if iscell(ref)
    masses = ref{1};
    if isequal(size(masses), [Kmax,1])
        masses = repmat(masses, 1, N);
    end
else
    if numel(ref) == 2
        ref = repmat(ref(:), 1, N);
    end
end
if nargin < 4
    Ks = repmat(Kmax, 1, N);
end

all_dists = [];
all_theta = [];
for nn = 1:N
    K = Ks(nn);
    pred = halos_pred(1:K,:,nn);
    true = halos_true(1:K,:,nn);
    [r_dists, idx] = calc_delta_r(pred, true);
    if iscell(ref)
        center = get_ref(true, masses(1:K,nn));
    else
        center = ref(:,nn);
    end
    theta = calc_theta(pred(idx,:), true, center);
    all_dists = [all_dists r_dists(:)'];
    all_theta = [all_theta theta(:)'];
end

c_dist = mean(all_dists);
c_cos = mean(cos(all_theta));
c_sin = mean(sin(all_theta));
c_angle = sqrt(c_cos^2 + c_sin^2);

cost = c_dist/1000 + c_angle;


function [r_dists, idx] = calc_delta_r(halos_pred, halos_true)
%CALC_DELTA_R Compute the scalar distance between predicted halo centers and the true halo centers.
%
%     [r_dist, idx] = calc_delta_r(x_pred, y_pred, x_true, y_true)
%
% Predictions are matched using best permutation.
%
% Inputs:
%      halos_pred Kx2
%      halos_true Kx2 
%
% Outputs:
%     r_dists Kx1 
%         idx 1xK 

% Iain Murray, December 2012

K = size(halos_pred, 1);
assert(size(halos_true, 1) == K);
matchings = perms(1:K);
M = size(matchings, 1);
assert(M == factorial(K));

best_mdist = Inf;
best_m = 0;
best_rdists = [];
for mm = 1:M
    halos_m = halos_pred(matchings(mm,:), :);
    r_dists = sqrt(sum((halos_m - halos_true).^2, 2));
    mdist = mean(r_dists);
    if mdist < best_mdist
        best_mdist = mdist;
        best_rdists = r_dists;
        best_m = mm;
    end
end
idx = matchings(best_m,:);
r_dists = best_rdists;


function theta = calc_theta(halos_pred, halos_true, ref)
%CALC_THETA angles for each halo used in dark world's metric
%
%     theta = calc_theta(halos_pred, halos_true, ref)
%
% Assumes halos have already been permuted to match them up.
%
% Inputs:
%      halos_pred Kx2 
%      halos_true Kx2 
%             ref 1x2 
%
% Outputs:
%          theta  Kx1 

% Direct port of provided python code. I would have thought this could be a lot
% neater using atan2.

x_true = halos_true(:, 1);
y_true = halos_true(:, 2);
x_pred = halos_pred(:, 1);
y_pred = halos_pred(:, 2);
x_ref = ref(1);
y_ref = ref(2);
assert(numel(ref) == 2);

K = size(halos_pred, 1);

% Special case code moved in here (hopefully correctly?), where it seems more sensible.
if K == 1
    if (x_pred - x_true) ~= 0
        psi = atan((y_pred-y_true)./(x_pred-x_true));
    else
        psi = 0;
    end
    theta = convert_to_360(psi, x_pred-x_true, y_pred-y_true);
    return
end

phi = zeros(K, 1);

psi = atan( (y_true-y_ref)./(x_true-x_ref) );
% BUG? indexing should be the same in all cases.
phi(x_true ~= x_ref) = atan(...
        (y_pred(x_true ~= x_pred) - y_true(x_true ~= x_pred)) ./...
            (x_pred(x_true ~= x_pred) - x_true(x_true ~= x_pred)) );
phi = convert_to_360(phi, x_pred-x_true, y_pred-y_true);
psi = convert_to_360(psi, x_true-x_ref, y_true-y_ref);
theta = phi - psi;
theta(theta < 0.0) = theta(theta < 0.0) + 2*pi;

function angle = convert_to_360(angle, x_in, y_in)
n = length(x_in);
for i = 1:n
    if x_in(i) < 0 && y_in(i) > 0
        angle(i) = angle(i)+pi;
    elseif x_in(i) < 0 && y_in(i) < 0
        angle(i) = angle(i)+pi;
    elseif x_in(i) > 0 && y_in(i) < 0
        angle(i) = angle(i)+2.0*pi;
    elseif x_in(i) == 0 && y_in(i) == 0
        angle(i) = 0;
    elseif x_in(i) == 0 && y_in(i) > 0
        angle(i) = pi/2;
    elseif x_in(i) < 0 && y_in(i) == 0
        angle(i) = pi;
    elseif x_in(i) == 0 && y_in(i) < 0
        angle(i) = 3.*pi/2;
    end
end

function ref = get_ref(halos, masses)
%GET_REF reference point: center of mass
%
%     ref = get_ref(halos, masses)
%
% Inputs:
%       halos Kx2 
%      masses Kx1 
%
% Outputs:
%        ref  1x2 

ref = sum(bsxfun(@times, halos, masses(:)/sum(masses)), 1);
