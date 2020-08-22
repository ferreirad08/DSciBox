function y = normpdf(x,mu,sigma)
if nargin < 2
    mu = 0;
end
if nargin < 3
    sigma = 1;
end
y = 1./(sigma.*sqrt(2.*pi))...
    .*exp(-1/2.*((x-mu)./sigma).^2);
end

function y = sigmoid(x)
y = 1./(1+exp(-x));
end
