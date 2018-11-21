% Used to look at raw data for individual training sky

% Iain Murray, December 2012

locs = load('locations');
refs = load('ref_points');

sky = ceil(rand()*100); % pick random sky
%sky = 84;

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
