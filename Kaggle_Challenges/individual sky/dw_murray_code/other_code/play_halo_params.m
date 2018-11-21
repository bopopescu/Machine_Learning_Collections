% numberHalos,x_ref,y_ref,halo_x1,halo_y1,halo_x2,halo_y2,halo_x3,halo_y3
summary_data = load('simple_data/Training_halos');

%for sky = 1:300
%for sky = 202
for blah = 1:12
    sky = ceil(rand*300);
    fprintf('doing sky %d\n', sky);
    K = summary_data(sky, 1);
    
    % x,y,e1,e2
    sky_data = load(sprintf('simple_data/Train_Skies/%d', sky));
    xx = sky_data(:, 1);
    yy = sky_data(:, 2);
    e1 = sky_data(:, 3);
    e2 = sky_data(:, 4);
    
    %%% Get true pos when first investigating params:
    halo_pos = zeros(K,2);
    for kk = 1:K
        halo_pos(kk, :) = summary_data(sky, 4+(kk-1)*2:5+(kk-1)*2);
    end
 
    box_width = 4200;
    N = 200;
    burn = 100;
    width = 2*box_width;
    logdist = @(halo_params) flex_ll_known_pos(halo_params, halo_pos, xx, yy, e1, e2);
    
    halo_params = repmat([350, 50, 0.2], K, 1);
    post_halo_params = slice_sample(N, burn, logdist, halo_params, width, false);
    post_halo_params = reshape(post_halo_params, K, 3, N);
    
    figure(blah); clf;
    title(sprintf('sky %d', sky));
    subplot(3,1,1);
    plot(squeeze(post_halo_params(:, 1, :))');
    ylabel('r thresh');
    subplot(3,1,2);
    plot(squeeze(post_halo_params(:, 2, :))');
    ylabel('r scale - "mass"');
    subplot(3,1,3);
    plot(squeeze(post_halo_params(:, 3, :))');
    ylabel('std boost');
end
