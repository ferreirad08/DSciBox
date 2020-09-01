classdef StandardData
%Standard Data
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    n
    mu_
    sigma_
end
methods
    function obj = StandardData(n)
        if nargin > 0
            obj.n = n;
        end
    end
    function obj = fit(obj,X)
        obj.mu_ = mean(X);
        obj.sigma_ = std(X,1);
    end
    function Xt = transform(obj,X)
        n_ = size(X,1);
        Xt = (X - repmat(obj.mu_,n_,1)) ./ repmat(obj.sigma_,n_,1);
    end
end
end
