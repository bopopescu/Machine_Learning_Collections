% Look for anisotropy. Creates plot of raw data in figure 3, and binned plots of
% e_tan for each of four quadrants around the halo in figure 4. I decided/hoped
% that the data was probably too noisy to get much advantage from modelling any
% anisotropies close to the halo.

% Apologies, this one script doesn't work in Octave (v3.2.4). There wasn't a
% quick fix: the format specifier could be removed from the error bars, but then
% they wouldn't be easily distinguishable.

% Iain Murray, December 2012

sky = ceil(rand*100);
%sky = 21;

N = 52982; % num galaxies in all skies
e_tan = zeros(N, 1);
r2 = zeros(N, 1);

locs = load('locations');

data = load(sprintf('sky/%d', sky));
nn = size(data, 1);
xp = locs(sky, 1);
yp = locs(sky, 2);
%xp = rand()*4200;
%yp = rand()*4200;
xx = data(:,1);
yy = data(:,2);
e1 = data(:,3);
e2 = data(:,4);
phi = atan((yy-yp)./(xx-xp));
e_tan = -(e1.*cos(2*phi) + e2.*cos(2*phi));
r2 = (xx-xp).^2 + (yy-yp).^2;

%figure(3); clf;
%plot(sqrt(r2), e_tan, '.')

figure(4); clf; hold on;
mx = max(sqrt(r2));
mI = 10;
hh = mx/mI;
plot(1:mI, zeros(1,mI), '-r');
for ii = 1:mI
    idx = (sqrt(r2)>((ii-1)*hh)) & (sqrt(r2)<(ii*hh));
    idx = idx & ((xx-xp)>0) & ((yy-yp)>0);
    errorbar(ii+randn()*0.1, mean(e_tan(idx)), std(e_tan(idx))/sqrt(sum(idx)), 'xb');
end
for ii = 1:mI
    idx = (sqrt(r2)>((ii-1)*hh)) & (sqrt(r2)<(ii*hh));
    idx = idx & ((xx-xp)>0) & ((yy-yp)<0);
    errorbar(ii+randn()*0.1, mean(e_tan(idx)), std(e_tan(idx))/sqrt(sum(idx)), '+m');
end
for ii = 1:mI
    idx = (sqrt(r2)>((ii-1)*hh)) & (sqrt(r2)<(ii*hh));
    idx = idx & ((xx-xp)<0) & ((yy-yp)<0);
    errorbar(ii+randn()*0.1, mean(e_tan(idx)), std(e_tan(idx))/sqrt(sum(idx)), 'or');
end
for ii = 1:mI
    idx = (sqrt(r2)>((ii-1)*hh)) & (sqrt(r2)<(ii*hh));
    idx = idx & ((xx-xp)<0) & ((yy-yp)>0);
    errorbar(ii+randn()*0.1, mean(e_tan(idx)), std(e_tan(idx))/sqrt(sum(idx)), '.k');
end


figure(3); clf;

data = load(sprintf('sky/%d', sky));
nn = size(data, 1);
xp = locs(sky, 1);
yp = locs(sky, 2);
%xp = rand()*4200;
%yp = rand()*4200;
xx = data(:,1);
yy = data(:,2);
e1 = data(:,3);
e2 = data(:,4);
phi = atan((yy-yp)./(xx-xp));
e_tan = -(e1.*cos(2*phi) + e2.*cos(2*phi));
r2 = (xx-xp).^2 + (yy-yp).^2;

clf; hold on;
plot(xp, yp, 'rx')
plot(xx, yy, 'b.')
theta = atan2(e2,e1)/2;
ww = 50;
for ii = 1:size(xx,1)
    plot([xx(ii)-ww*cos(theta(ii)) xx(ii)+ww*cos(theta(ii))], [yy(ii)-ww*sin(theta(ii)) yy(ii)+ww*sin(theta(ii))], 'b-')
end
axis square
