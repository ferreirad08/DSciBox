function Q = quantile(X,p)
%Author: David Ferreira - Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%Quantile Analysis
%
%Syntax
%1. Q = quantile(X,p)
%
%Description 
%1. Calculate the quantiles of a vector or matrix data based on linear regression.
%
%Examples
%1.
%     v = [10 1 2 3 4 7];
%     p = [0.25 0.50 0.75];
%     Q = quantile(v,p)
%     Q = 
%         2.2500
%         3.5000
%         6.2500
%
%2.
%     X = [1 2; 2 5; 3 6; 4 10; 7 11; 10 13];
%     Q = quantile(X,p)
%     Q = 
%         2.2500    5.2500
%         3.5000    8.0000
%         6.2500   10.7500

if isrow(X), X = X'; end
if isrow(p), p = p'; end

X = sort(X);
[n,m] = size(X);

X(n+1,:) = 0;

i = (n-1)*p+1;
f = floor(i);
Q = X(f,:) + (X(f+1,:)-X(f,:)).*repmat(i-f,1,m);
end
