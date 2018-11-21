function samples = slice_sample(N, burn, logdist, xx, widths, step_out, verbose, varargin)
%SLICE_SAMPLE simple axis-aligned implementation of slice sampling for vectors
%
%     samples = slice_sample(N, burn, logdist, xx, widths, step_out, varargin)
%
% Inputs:
%             N  1x1  Number of samples to gather
%          burn  1x1  after burning period of this length
%       logdist  @fn  function logprobstar = logdist(xx, varargin{:})
%            xx  Dx1  initial state (or array with D elements)
%        widths  Dx1  or 1x1, step sizes for slice sampling
%      step_out bool  set to true if widths may sometimes be far too small
%       verbose  1x1  default 1, set to 2 to also get stepping
%      varargin   -   any extra arguments are passed on to logdist
%
% Outputs:
%      samples  DxN   samples stored in columns (regardless of original shape)
%
% Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010, December 2012 added verbose option.
% See Pseudo-code in David MacKay's text book p375

% startup stuff
D = numel(xx);
samples = zeros(D, N);
if numel(widths) == 1
    widths = repmat(widths, D, 1);
end
log_Px = feval(logdist, xx, varargin{:});
if nargin < 7
    verbose = 1;
end

% Main loop
for ii = 1:(N+burn)
    if verbose > 0
        fprintf('Iteration %d                 \r', ii - burn);
    end
    log_uprime = log(rand) + log_Px;

    % Sweep through axes (simplest thing)
    for dd = 1:D
        x_l = xx;
        x_r = xx;
        xprime = xx;

        % Create a horizontal interval (x_l, x_r) enclosing xx
        rr = rand;
        x_l(dd) = xx(dd) - rr*widths(dd);
        x_r(dd) = xx(dd) + (1-rr)*widths(dd);
        if step_out
            % Typo in early editions of book. Book said compare to u, but it should say u'
            while (feval(logdist, x_l, varargin{:}) > log_uprime)
                x_l(dd) = x_l(dd) - widths(dd);
            end
            while (feval(logdist, x_r, varargin{:}) > log_uprime)
                x_r(dd) = x_r(dd) + widths(dd);
            end
        end

        % Inner loop:
        % Propose xprimes and shrink interval until good one found
        zz = 0;
        while 1
            zz = zz + 1;
            if verbose > 1
                fprintf('Iteration %d   Step %d       \r', ii - burn, zz);
            end
            xprime(dd) = rand()*(x_r(dd) - x_l(dd)) + x_l(dd);
            log_Px = feval(logdist, xprime, varargin{:});
            if log_Px > log_uprime
                break % this is the only way to leave the while loop
            else
                % Shrink in
                if xprime(dd) > xx(dd)
                    x_r(dd) = xprime(dd);
                elseif xprime(dd) < xx(dd)
                    x_l(dd) = xprime(dd);
                else
                    error('BUG DETECTED: Shrunk to current position and still not acceptable.');
                end
            end
        end
        xx(dd) = xprime(dd);
    end

    % Record samples
    if ii > burn
        samples(:, ii - burn) = xx(:);
    end
end
if verbose > 0
    fprintf('\n');
end

