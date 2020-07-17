classdef MinMaxNormalizer
%Min-Max Normalizer
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    n
    min_
    max_
end
methods
    function obj = MinMaxNormalizer(n)
        if nargin > 0
            obj.n = n;
        end
    end
    function obj = fit(obj,X)
        obj.min_ = min(X);
        obj.max_ = max(X);
    end
    function Xn = transform(obj,X)
        n_ = size(X,1);
        Xn = (X-repmat(obj.min_,n_,1))...
            ./(repmat(obj.max_,n_,1)-repmat(obj.min_,n_,1));
    end
end
end
