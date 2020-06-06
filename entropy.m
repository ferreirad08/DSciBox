function h = entropy(Y)
    % Calculate the probabilities
    p = histc(Y,unique(Y))/numel(Y);
    h = sum(-p.*log2(p));
end
