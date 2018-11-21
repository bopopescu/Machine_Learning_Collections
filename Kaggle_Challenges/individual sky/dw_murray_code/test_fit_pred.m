% After drawing posterior samples, try to optimize expected cost(ish).

for sky = 1:120
    if exist(sprintf('tpred/%d.mat', sky), 'file')
        continue
    end
    system(sprintf('touch tpred/%d.mat', sky));
    fprintf('sky = %d\n', sky);
    ws = load(sprintf('tsamples/sky%d.mat', sky));
    post_halos = ws.post_halos; % Kx5xS, K X (x,y,r_0,inv_m,sigma_inc) x num_samples
    K = size(post_halos, 1);
    
    masses = 1./permute(post_halos(:,4,:), [1 3 2]); % KxS
    halos_true = post_halos(:,1:2,:);
    halos_pred = median(halos_true(:,1:2,1:100), 3);
    
    val_sub = 1:1000;
    h_val = halos_true(:,:,val_sub);
    m_val = masses(:,val_sub);
    best_cost = dw_metric(halos_pred, h_val, {m_val});
    best_halos = halos_pred;
    for tt = 1:200
        fprintf('tt = %d\r', tt);
        n_subset = 10;
        subset = ceil(rand(1, n_subset)*size(halos_true, 3));
        h_sub = halos_true(:,:,subset);
        m_sub = masses(:,subset);
    
        iters = 10; 
        verbose = false;
        costfn = @(halos_pred) dw_metric(halos_pred, h_sub, {m_sub});
        min_step = 100;
        max_step = 5000;
        halos_pred = hack_opt(iters, costfn, halos_pred, verbose, max_step, min_step);
        cost = dw_metric(halos_pred, h_val, {m_val});
        if cost < best_cost
            best_halos = halos_pred;
            best_cost = cost
        else
            halos_pred = best_halos;
        end
    end
    fprintf('\n');
    
    save(sprintf('tpred/%d.mat', sky), 'halos_pred');
end

