classdef InformationGain
%Information Gain (IG)
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
    n_features
    gain
    indexes
end
methods
    function obj = InformationGain(n_features)
        if nargin > 0
            obj.n_features = n_features;
        end
    end
    function obj = fit(obj,X,Y)
        [m,n] = size(X);
        obj.gain = ones(1,n)*dsb_utilities.entropy(Y);
        for i = 1:n
            [~,~,feature] = unique(X(:,i));
            for j = 1:max(feature)
                p = histc(feature(feature==j),j)/m;
                obj.gain(i) = obj.gain(i) - p*dsb_utilities.entropy(Y(feature==j));
            end
        end

        [obj.gain,obj.indexes] = sort(obj.gain,'descend');
    end
    function Xt = feature_selection(obj,X)
        Xt = X(:,obj.indexes(1:obj.n_features));
    end
end
end
