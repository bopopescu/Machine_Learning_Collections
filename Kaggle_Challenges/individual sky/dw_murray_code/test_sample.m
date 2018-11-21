summary_data = load('simple_data/Test_halos');

for sky = 1:120
    if exist(sprintf('tsamples/sky%d.mat', sky), 'file')
        continue
    end
    system(sprintf('touch tsamples/sky%d.mat', sky));
    fprintf('doing sky %d\n', sky);
    K = summary_data(sky, 1);
    
    % x,y,e1,e2
    sky_data = load(sprintf('simple_data/Test_Skies/%d', sky));
    xx = sky_data(:, 1);
    yy = sky_data(:, 2);
    e1 = sky_data(:, 3);
    e2 = sky_data(:, 4);
    
    % Slice sampling parameters
    N = 1000;
    burn = 100;
    box_width = 4200;
    %width = repmat([2*box_width, 2*box_width, 2*1000, 2*300, 2*0.3], K, 1); % Used in later submissions
    width = repmat([2*box_width, 2*box_width, 2*1000, 2*300, 0.001], K, 1);
    logdist = @(halos) flex_ll(halos, xx, yy, e1, e2);
    
    % Initialization
    best_halos = [];
    best_ll = -Inf;
    fprintf('   Initializing\n');
    while true
        for i = 1:4000
            halos = bsxfun(@times, rand(K,5), [box_width, box_width, 1000, 200, 0]);
            ll = logdist(halos);
            if ll > best_ll
                best_halos = halos;
                best_ll = ll;
            end
            % I thought about evolving each of the starting points before
            % considering, but didn't:
            %halos = reshape(slice_sample(1, 5, logdist, halos, width, false, 0), K, 5);
            %if ll > best_ll
            %    best_halos = halos;
            %    best_ll = ll;
            %end
        end
        if best_ll == -Inf
            fprintf('Warning: finding it hard to find initialization, sky=%d\n', sky);
        else
            break;
        end
    end
    halos = best_halos;
    
    % Main work done here:
    post_halos= slice_sample(N, burn, logdist, halos, width, false);
    post_halos = reshape(post_halos, K, 5, N);
    
    save(sprintf('tsamples/sky%d.mat', sky), 'post_halos');
end
