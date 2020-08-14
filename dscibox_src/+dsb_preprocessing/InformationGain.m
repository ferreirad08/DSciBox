classdef InformationGain
%Feature Selection Based on Information Gain (IG)
%
% SYNTAX
% 1. ig = dsb_preprocessing.InformationGain(n_features)
%    ig = ig.fit(X,Y)
%    Xr = ig.feature_selection(X)
%
% DESCRIPTION
% 1. Discrete the continuous variables for each column of a matrix based on quantiles.
%
% X is a M-by-N matrix, with M instances of N features. 
% Y is a M-by-1 matrix, with respective M labels to each training instance. 
% n_features is the number of features in Xr (reduced set).
%
% EXAMPLE
% 1.
%      X = [5.1  3.5  1.4  0.2;
%           4.9  3.0  1.4  0.2;
%           4.7  3.2  1.3  0.2;
%           6.3  3.3  4.7  1.6;
%           4.9  2.4  3.3  1.0];
%      Y = [1; 1; 1; 2; 2; 2];
%      n_features = 2;
%      ig = dsb_preprocessing.InformationGain(n_features)
%      ig = ig.fit(X,Y)
%      Xr = ig.feature_selection(X)
%      Xr =
%          3.5  1.4
%          3.0  1.4
%          3.2  1.3
%          3.3  4.7
%          2.4  3.3
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
        obj.gain = ones(1,n)*dsb_utils.entropy(Y);
        for i = 1:n
            [~,~,feature] = unique(X(:,i));
            for j = 1:max(feature)
                p = histc(feature(feature==j),j)/m;
                obj.gain(i) = obj.gain(i) - p*dsb_utils.entropy(Y(feature==j));
            end
        end

        [obj.gain,obj.indexes] = sort(obj.gain,'descend');
    end
    function Xt = feature_selection(obj,X)
        Xt = X(:,obj.indexes(1:obj.n_features));
    end
end
end
