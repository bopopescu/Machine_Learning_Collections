% This code binned up radial distance away from each halo, and looked at the
% mean (with standard error, blue error bars) of the e_tan quantity computed in
% the example maximum likelihood code. On the basis of this code (and
% interactive tweaks of it), I decided to use "force" = 1/( max(r,r0)/scale ),
% where scale is proportional to 1/mass.
%
% The red +'s show the standard deviation of e_tan, which I took as a hint to
% increase noise inside the core. But this decision could have been entirely
% spurious. I should have simulated data from my model and seen if this increase
% in standard deviation occurs naturally of an artifact of my plotting
% procedure.

% Iain Murray, December 2012

N = 52982; % num galaxies in all skies
e_tan = zeros(N, 1);
r2 = zeros(N, 1);

locs = load('locations');

idx = 1;
for ii = 1:100 % To combine stats from all skies
%for ii = ceil(rand*100) % To look at a random sky
%for ii = 47 % To look at a particular single sky
    data = load(sprintf('sky/%d', ii));
    nn = size(data, 1);
    xp = locs(ii, 1);
    yp = locs(ii, 2);
    %xp = rand()*4200; % Used as a control
    %yp = rand()*4200;
    xx = data(:,1);
    yy = data(:,2);
    e1 = data(:,3);
    e2 = data(:,4);
    phi = atan((yy-yp)./(xx-xp));
    e_tan(idx:idx+nn-1, :) = -(e1.*cos(2*phi) + e2.*cos(2*phi));
    r2(idx:idx+nn-1, :) = (xx-xp).^2 + (yy-yp).^2;
    idx = idx + nn;
end
e_tan = e_tan(1:idx-1,:);
r2 = r2(1:idx-1,:);

%figure(3); clf;
%plot(sqrt(r2), e_tan, '.')

figure(2); clf; hold on;
mx = max(sqrt(r2));
mI = 50;
hh = mx/mI;
plot(1:mI, zeros(1,mI), '-r');
for ii = 1:mI
    idx = (sqrt(r2)>((ii-1)*hh)) & (sqrt(r2)<(ii*hh));
    %plot(ii, mean(e_tan(idx)), 'xb');
    %errorbar(ii, mean(e_tan(idx)), std(e_tan(idx))/sqrt(sum(idx)), 'xb'); % doesn't work in Octave 3.2.4
    errorbar(ii, mean(e_tan(idx)), std(e_tan(idx))/sqrt(sum(idx)));
    plot(ii, std(e_tan(idx)), '+r');
    plot(ii, 1/(hh*ii/50), 'ok');
end
