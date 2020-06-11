function [Xt,Q] = binning(X,arg2)
%Author: David Ferreira
%PhD student in Electrical Engineering
%Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%Quantile Binning Transformation
%
%Syntax
%1. [Xt,Q] = binning(X,n_bins)
%2. Xt = binning(X,Q)
%
%Description
%1. Discrete the continuous variables for each column of a matrix based on quantiles.
%
%X is a M-by-N matrix with continuous variables in each column.
%n_bins is the number of groupings (n_bins > 2).
%
%Examples
%1.
%     X = [16     2;
%           5    11;
%           9     7;
%           4    14];
%     n_bins = 3;
%     [Xt,Q] = binning(X,n_bins)
%     Xt =
%          2     0
%          1     2
%          2     1
%          0     2
%     Q =
%          5     7
%          9    11
%
%2.
%     X2 = [ 3    13;
%           10     8;
%            6    12;
%           15     1];
%     X2t = binning(X2,Q)
%     X2t =
%          0     2
%          2     1
%          1     2
%          2     0

if isscalar(arg2)
    n_bins = arg2;
    p = (1:n_bins-1)/n_bins;
    Q = quantile(X,p);
else
    Q = arg2;
    n_bins = size(Q,1)+1;
end

[m,n] = size(X);
Xt = zeros(m,n);
for i = 1:n
    Xt(:,i) = sum(repmat(X(:,i)',n_bins-1,1)...
        >=repmat(Q(:,i),1,m));
end
end
