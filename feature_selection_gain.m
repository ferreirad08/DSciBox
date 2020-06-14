function [Xt,indexes,gain] = feature_selection_gain(X,Y,k)
%Author: David Ferreira - Federal University of Amazonas
%PhD student in Electrical Engineering
%Contact: ferreirad08@gmail.com
%
%Information Gain
%
%Syntax
%1. [Xt,indexes,g] = feature_selection_gain(X,Y,k)

[n_samples,n_features] = size(X);
gain = ones(1,n_features)*entropy(Y);
for i = 1:n_features
    [~,~,feature] = unique(X(:,i));
    for j = 1:max(feature)
        p = histc(feature(feature==j),j)/n_samples;
        gain(i) = gain(i) - p*entropy(Y(feature==j));
    end
end

[gain,indexes] = sort(gain,'descend');
Xt = X(:,indexes(1:k));
end
