function h = entropy(Y)
    p = histc(Y,unique(Y))/numel(Y);
    h = sum(-p.*log2(p));
end
