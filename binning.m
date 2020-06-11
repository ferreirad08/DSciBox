function [Xt,Q] = binning(X,arg2)
%Author: David Ferreira - Federal University of Amazonas
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
%     v = [10 1 2 3 4 7];
%     n_bins = 3;
%     vt = binning(v',n_bins)
%     vt =
%         2
%         0
%         0
%         1
%         1
%         2
%
%2.
%     X = [-2, 1, -4,   -1;
%          -1, 2, -3, -0.5;
%           0, 3, -2,  0.5;
%           1, 4, -1,    2];
%     n_bins = 3;
%     Xt = binning(X,n_bins)
%     Xt =
%         0     0     0     0
%         1     1     1     1
%         2     2     2     2
%         2     2     2     2

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
