function y = raylpdf(x,b)
if nargin < 2
    b = mode(x); % Most frequent value
end
y = x/(b^2).*exp(-x.^2/(2*b^2));
end
