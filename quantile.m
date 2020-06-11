function Q = quantile(X,p)
%Author: David Ferreira
%PhD student in Electrical Engineering
%Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%Quantile Analysis
%
%Syntax
%1. Q = quantile(X,p)
%
%Description 
%1. Calculate the quantiles for each column of a matrix based on linear regression.
%
%Examples
%1.
%     X = [ 1  2;
%           2  5;
%           3  6;
%           4 10;
%           7 11;
%          10 13];
%     Q = quantile(X,p)
%     Q = 
%         2.2500    5.2500
%         3.5000    8.0000
%         6.2500   10.7500

if ~isrow(X), X = sort(X); end
if isrow(p), p = p'; end

[n,m] = size(X);
X(n+1,:) = 0;

i = (n-1)*p+1;
f = floor(i);
Q = X(f,:) + (X(f+1,:)-X(f,:))...
    .*repmat(i-f,1,m);
end
