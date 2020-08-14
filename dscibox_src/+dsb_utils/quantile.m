function Q = quantile(X,p)
%Quantile Analysis
%
% SYNTAX
% 1. Q = dsb_utils.quantile(X,p)
%
% DESCRIPTION
% 1. Calculate the quantiles for each column of a matrix based on linear regression.
%
% EXAMPLE
% 1.
%      X = [10    2;
%            2    5;
%            3    6;
%            4   13;
%            7   10;
%            1   11];
%      p = [0.25 0.50 0.75];
%      Q = dsb_utils.quantile(X,p)
%      Q = 
%          2.2500    5.2500
%          3.5000    8.0000
%          6.2500   10.7500
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

if ~isrow(X)
    X = sort(X);
end
if isrow(p)
    p = p';
end
[n,m] = size(X);
X(n+1,:) = 0;
i = (n-1)*p+1;
f = floor(i);
Q = X(f,:) + (X(f+1,:)-X(f,:))...
    .*repmat(i-f,1,m);
end
