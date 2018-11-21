function ll = flex_ll(halos, xx, yy, e1, e2)
%LOG_LIKELIHOOD of halos (and other params) given a field of galaxies.
%
%     ll = log_likelihood(halos, xx, yy, e1, e2)
%
% Inputs:
%     halos Kx5 (x,y,r_0,inv_m,sigma_inc) K halos: x-pos, y-pos, cut-off radius, 1/"mass", boost to noise within cut-off
%        xx Nx1 x-positions of Galaxies
%        yy Nx1 y-positions of Galaxies
%        e1 Nx1 x-axis ellipticities of Galaxies
%        e2 Nx1 45-degree ellipticities of Galaxies
%
% Outputs:
%       ll  1x1 

% Iain Murray, December 2012

% This bit actually encodes *prior* constraints:
bad = (min(halos(:,1)) < 0) || (max(halos(:,1)) > 4200);
bad = bad || (min(halos(:,2)) < 0) || (max(halos(:,2)) > 4200);
bad = bad || (min(halos(:,3)) < 0) || (max(halos(:,3)) > 1000);
bad = bad || (min(halos(:,4)) < 0) || (max(halos(:,4)) > 300);
bad = bad || (min(halos(:,5)) < 0) || (max(halos(:,5)) > 0.01); % TODO should I allow extra variance?
%bad = bad || (min(halos(:,5)) < 0) || (max(halos(:,5)) > 0.3); % used instead of line above in later runs
if bad
    ll = -Inf;
    return;
end

e_std = 0.22;

N = length(xx);
e1_force = zeros(N, 1);
e2_force = zeros(N, 1);
pred_var = repmat(e_std.^2, N, 1);
for kk = 1:size(halos, 1)
    x0 = halos(kk, 1);
    y0 = halos(kk, 2);
    r_thresh = halos(kk, 3);
    scale = halos(kk, 4);
    core_std = e_std + halos(kk, 5);
    angle_wrt_centre = atan((yy-y0)./(xx-x0));
    r_from_halo = sqrt((xx-x0).^2+(yy-y0).^2);
    force = 1 ./ (max(r_thresh, r_from_halo) / scale);
    e1_force = e1_force - force .* cos(2*angle_wrt_centre);
    e2_force = e2_force - force .* sin(2*angle_wrt_centre);
    pred_var(r_from_halo < r_thresh) = max(pred_var(r_from_halo < r_thresh), core_std^2);
end 
ll = sum(-0.5*((e1-e1_force).^2 + (e2-e2_force).^2)./pred_var - log(2*pi*pred_var));

