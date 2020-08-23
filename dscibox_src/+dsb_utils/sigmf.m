function y = sigmf(x,params)
if nargin < 2
    params = [1 0];
end
y = 1./(1+exp(-params(1)...
    .*(x-params(2))));
end
