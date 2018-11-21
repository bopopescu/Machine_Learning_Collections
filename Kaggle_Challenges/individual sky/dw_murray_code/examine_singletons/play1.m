% This code was used to quickly eye-ball the marginal distribution of e1 and e2,
% and the joint marginal distribution of (e1,e2). I didn't do any careful
% testing: it seemed so likely that they were independent Gaussians (given that
% I knew the data was synthetic) that I just assumed that from then on.

% Iain Murray, December 2012

e1 = [];
e2 = [];
for ii = 1:100
    xx = load(sprintf('sky/%d', ii));
    e1 = [e1; xx(:,3)];
    e2 = [e2; xx(:,4)];
end
figure(1); clf;
hist(e1/std(e1),100)
figure(2); clf;
hist(e2/std(e2),100)
figure(3); clf;
plot(e1(1:3000),e2(1:3000),'.')

