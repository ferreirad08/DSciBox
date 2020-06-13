function [Xt,indexes,g] = gain(X,Y,k)
% Gain of information

[n_samples,n_features] = size(X);
g = ones(1,n_features)*entropy(Y);
for i = 1:n_features
    [~,~,feature] = unique(X(:,i));
    for j = 1:max(feature)
        p = histc(feature(feature==j),j)/n_samples;
        g(i) = g(i) - p*entropy(Y(feature==j));
    end
end

[g,indexes] = sort(g,'descend');
Xt = X(:,indexes(1:k));
end
