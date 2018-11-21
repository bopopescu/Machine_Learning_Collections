% Pick a posterior sample from each sky and write out as a fantasy dataset so I
% can see what my predictions would score against it.

fid = fopen('test_fantasy.csvTraining_halos.csv', 'w');
fprintf(fid, 'SkyId,numberHalos,x_ref,y_ref,halo_x1,halo_y1,halo_x2,halo_y2,halo_x3,halo_y3\n');

for sky = 1:120
    ws = load(sprintf('tsamples/sky%d.mat', sky));
    post_halos = ws.post_halos; % Kx5xS, K X (x,y,r_0,inv_m,sigma_inc) x num_samples
    id = ceil(rand()*size(post_halos,3));
    halos = post_halos(:,1:2,id);
    K = size(halos,1);
    positions = halos';
    positions = positions(:)';
    
    masses = 1./permute(post_halos(:,4,end), [1 3 2]); % Kx1
    ref = sum(bsxfun(@times, halos, masses(:)/sum(masses)), 1);

    fprintf(fid, 'Sky%d,%d,%g,%g,%g,%g,%g,%g,%g,%g\n', sky, K, ref(:)', [positions zeros(1, 2*(3-K))]);
end

fclose(fid);
