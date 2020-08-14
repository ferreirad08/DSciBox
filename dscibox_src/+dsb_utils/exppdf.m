function y = exppdf(x,mu)
if nargin < 2
    mu = 1;
end
lambda = 1./mu; % Rate Parameter
y = lambda.*exp(-lambda.*x);
end
