function ll = flex_ll_known_pos(halo_params, halo_pos, xx, yy, e1, e2)
%LOG_LIKELIHOOD of halos (and other params) given a field of galaxies.
%
%     ll = log_likelihood(halos, xx, yy, e1, e2)
%
% Inputs:
%     halo_params Kx3 (r_0,inv_m,sigma_inc) K halos: cut-off radius, 1/"mass", boost to noise within cut-off
%        halo_pos Kx2 (x,y) K halos: x-pos, y-pos
%              xx Nx1 x-positions of Galaxies
%              yy Nx1 y-positions of Galaxies
%              e1 Nx1 x-axis ellipticities of Galaxies
%              e2 Nx1 45-degree ellipticities of Galaxies
%
% Outputs:
%       ll  1x1 

% Iain Murray, December 2012

halos = [halo_pos, halo_params];
ll = flex_ll_wider_cutoff(halos, xx, yy, e1, e2);
