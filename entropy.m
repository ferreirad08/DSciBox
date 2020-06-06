function e = entropy(Y)
    p = histc(Y,unique(Y))/numel(Y);
    e = sum(-p.*log2(p));
end
