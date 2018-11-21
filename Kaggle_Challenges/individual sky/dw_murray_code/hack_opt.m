function xx = hack_opt(N, costfn, xx, verbose, max_step, min_step)
%HACK_OPT 
%
%     xx = hack_opt(N, costfn, xx, verbose)
%
% Inputs:
%              N  1x1 number of iterations
%         costfn  @fn
%             xx  KxD K points in D-dimensions
%        verbose bool
%
% Outputs:
%            xx   KxD 

% Iain Murray, December 2012

if nargin < 4
    verbose = true;
end
if nargin < 5
    max_step = 5000;
end
if nargin < 6
    min_step = 100;
end

[K, D] = size(xx);

best_cost = costfn(xx);

for ii = 1:N
    if verbose > 0
        fprintf('Iteration %d\r', ii);
    end

    % Sweep through points
    for kk = 1:K
        dir = randn(1, D);
        dir = dir / sqrt(dir(:)'*dir(:));

        x_l = xx;
        x_r = xx;
        xprime = xx;

        x_r(kk,:) = xx(kk,:) + max_step*dir; % FIXME should be x_r(kk,:) = max_step*dir; -- but that's not what I used in submissions

        % Inner loop:
        % Propose xprimes and shrink interval until good one found or give up
        while 1
            uu = rand();
            xprime(kk,:) = uu*x_r(kk,:) + xx(kk,:);
            cost = costfn(xprime);
            if cost < best_cost
                xx(kk,:) = xprime(kk,:);
                best_cost = cost;
                break
            else
                % Shrink in
                x_r(kk,:) = uu*x_r(kk,:);
                if sqrt(sum((x_r(kk,:)).^2)) < min_step
                    break
                end
            end
        end
    end
end
if verbose > 0
    fprintf('\n');
end

