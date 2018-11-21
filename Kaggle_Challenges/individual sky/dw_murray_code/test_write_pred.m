fid = fopen('test_stab.csv', 'w');
fprintf(fid, 'SkyId,pred_x1,pred_y1,pred_x2,pred_y2,pred_x3,pred_y3\n');

for sky = 1:120
    load(sprintf('tpred/%d.mat', sky)); % loads halos_pred, Kx2
    guess = halos_pred';
    K = size(halos_pred, 1);
    fprintf(fid, 'Sky%d,%g,%g,%g,%g,%g,%g\n', sky, [guess(:)' zeros(1, 2*(3-K))]);
end

fclose(fid);
