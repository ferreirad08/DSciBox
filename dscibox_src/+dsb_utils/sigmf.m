function y = sigmf(x,params)
if nargin < 2
    params = [1, 0];
end
a = params(1);
c = params(2);
y = 1./(1+exp(-a.*(x-c)));
end
