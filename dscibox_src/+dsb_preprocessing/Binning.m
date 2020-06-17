classdef Binning
%Quantile Binning Transformation
%
% SYNTAX
% 1. [Xt,Q] = dsb_preprocessing.Binning(X,n_bins)
% 2. Xt = binning(X,Q)
%
% DESCRIPTION
% 1. Discrete the continuous variables for each column of a matrix based on quantiles.
%
% X is a M-by-N matrix with continuous variables in each column.
% n_bins is the number of groupings (n_bins > 2).
%
% EXAMPLE
% 1.
%      X = [16     2;
%            5    11;
%            9     7;
%            4    14];
%      n_bins = 3;
%      [Xt,Q] = dsb_preprocessing.Binning(X,n_bins)
%      Xt =
%           2     0
%           1     2
%           2     1
%           0     2
%      Q =
%           5     7
%           9    11
%
% 2.
%      X2 = [ 3    13;
%            10     8;
%             6    12;
%            15     1];
%      X2t = dsb_preprocessing.Binning(X2,Q)
%      X2t =
%           0     2
%           2     1
%           1     2
%           2     0
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    n_bins
    Q
end
methods
    function obj = Binning(n_bins)
        if nargin > 0
            obj.n_bins = n_bins;
        end
    end
    function obj = fit(obj,X)
        p = (1:obj.n_bins-1)/obj.n_bins;
        obj.Q = quantile(X,p);
    end
    function Xt = transform(obj,X)
        [m,n] = size(X);
        Xt = zeros(m,n);
        for i = 1:n
            Xt(:,i) = sum(repmat(X(:,i)',obj.n_bins-1,1)...
                >=repmat(obj.Q(:,i),1,m));
        end
    end
end
end
