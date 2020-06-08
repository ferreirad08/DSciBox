function e = entropy(Y)
    % Calculate the probabilities
    p = histc(Y,unique(Y))/numel(Y);
    % Calculate the entropy
    e = -sum(p.*log2(p));
end
