function Xn = MinMaxNormalizer(X)
Xn = (X-min(X))./(max(X)-min(X));
end
