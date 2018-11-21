fid = fopen('test_final_merge.csv', 'w');
fprintf(fid, 'SkyId,pred_x1,pred_y1,pred_x2,pred_y2,pred_x3,pred_y3\n');

num_skies = 120;

savings = zeros(1, num_skies);

for sky = num_skies:-1:1
    samps = {'tsamples2', 'tsamples2.1st', 'tsamples2.2nd', 'tsamples2.4th', 'tsamples3', 'tsamples3.2nd', 'tsamples3.3rd', 'tsamples3.4th'};
    post_halos = [];
    for ii = 1:length(samps)
        ws = load(sprintf('%s/sky%d.mat', samps{ii}, sky));
        post_halos = cat(3, post_halos, ws.post_halos); % Kx5xS, K X (x,y,r_0,inv_m,sigma_inc) x num_samples
    end
    masses = 1./permute(post_halos(:,4,:), [1 3 2]); % KxS
    halos_true = post_halos(:,1:2,:);
    h_val = halos_true;
    m_val = masses;

    preds = {'tpred3', 'tpred3.2nd', 'tpred3.big1', 'tpred3.big2', 'tpred3.big3', 'tpred3.big4', 'tpred3.3rd', 'tpred3.4th', 'tpred_pop', 'tpred_popb', 'tpred_popc', 'tpred_popd'};
    best_halos = [];
    best_cost = Inf;
    worst_cost = -Inf;
    costs = zeros(1, length(preds));
    for ii = 1:length(preds)
        clear halos_pred
        try
            load(sprintf('%s/%d.mat', preds{ii}, sky)); % loads halos_pred, Kx2
        catch
            fprintf('Skipping %s/%d.mat\n', preds{ii}, sky);
            continue
        end
        cost_file = sprintf('%s/%d_cost_final.mat', preds{ii}, sky);
        if ~exist(cost_file, 'file')
            cost = dw_metric(halos_pred, h_val, {m_val});
            save(cost_file, 'cost');
        else
            load(cost_file);
        end
        if (cost > worst_cost) && (ii <= 4)
            worst_cost = cost;
        end
        if cost < best_cost
            best_cost = cost;
            best_halos = halos_pred;
        end
        costs(ii) = cost;
    end
    guess = best_halos';
    savings(sky) = worst_cost - best_cost;
    fprintf('%g saved on sky %d, costs:', savings(sky), sky);
    fprintf(' %g', costs);
    fprintf('\n');

    K = size(best_halos, 1);
    fprintf(fid, 'Sky%d,%g,%g,%g,%g,%g,%g\n', sky, [guess(:)' zeros(1, 2*(3-K))]);
end

fprintf('Mean saving: %g\n', mean(savings));

fclose(fid);
