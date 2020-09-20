classdef kNNeighborsRegressor
%k-Nearest Neighbors Regressor
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% D.Sc. student in Electrical Engineering from the Federal University of
% Amazonas 
% e-mail: ferreirad08@gmail.com

properties
    k = 5
    metric = 'euclidean' % distance metric
    weights = 'uniform'  % weight function
    X
    Y
end
methods
    function obj = kNNeighborsRegressor(k,metric,weights)
        if nargin > 0
            obj.k = k;
        end
        if nargin > 1
            obj.metric = metric;
        end
        if nargin > 2
            obj.weights = weights;
        end
    end
    function obj = fit(obj,X,Y)
        obj.X = X;
        obj.Y = Y;
    end
    function Ypred = predict(obj,Xnew)
        dist = dsb_utils.cdist(obj.X,Xnew,obj.metric); % distances
        P = size(Xnew,1); % number of tests
        Ypred = zeros(1,P);
        if strcmp(obj.weights,'uniform')
            [~,ind] = sort(dist);
            for i = 1:P
                Ypred(i) = mean(obj.Y(ind(1:obj.k,i)));
            end
        elseif strcmp(obj.weights,'distance')
            [dist,ind] = sort(dist);
            w = 1./dist; % weights
            for i = 1:P
                Ypred(i) = sum(obj.Y(ind(1:obj.k,i)).*w(1:obj.k,i)')...
                    /sum(w(1:obj.k,i));
            end
        end
    end
end
end
