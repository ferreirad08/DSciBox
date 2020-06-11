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
%     X = [16     2     3    13;
%           5    11    10     8;
%           9     7     6    12;
%           4    14    15     1];
%     n_bins = 3;
%     [Xt,Q] = binning(X,n_bins)
%     Xt =
%          2     0     0     2
%          1     2     2     1
%          2     1     1     2
%          0     2     2     0
%     Q =
%          5     7     6     8
%          9    11    10    12
%
%2.
%     X2 = [21.3333    6.6667   12.0000    5.3333;
%            2.6667   14.6667    9.3333   18.6667;
%            4.0000   13.3333    8.0000   20.0000;
%           17.3333   10.6667   16.0000    1.3333];
%     X2t = binning(X2,Q)
%     X2t =
%          2     0     2     0
%          0     2     1     2
%          0     2     1     2
%          2     1     2     0

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
