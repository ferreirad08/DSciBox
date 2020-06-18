classdef kNNeighbors
%k-Nearest Neighbors (kNN)
%
% SYNTAX
% 1. mdl = dsb_predictors.kNNeighbors(k,p_norm)
%    mdl = mdl.fit(X,Y)
%    Ypred = mdl.predict(Xnew)
% 2. [Xnearest,Ynearest,distances] = mdl.find(Xnew(1,:))
%
% DESCRIPTION
% 1. Returns the estimated labels of one or multiple test instances.
% 2. Returns the values of the features, labels and distances of the k nearest instances to a new instance.
%
% X is a M-by-N matrix, with M instances of N features. 
% Y is a M-by-1 matrix, with respective M labels to each training instance. 
% Xnew is a P-by-N matrix, with P instances of N features to be classified.
% k is a scalar with the number of nearest neighbors selected.
% p_norm is the power parameter for the distance metric.
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    k = 5
    p_norm = 2
    C
    X
    Y
end
methods
    function obj = kNNeighbors(k,p_norm)
        if nargin > 0
            obj.k = k;
        end
        if nargin > 1
            obj.p_norm = p_norm;
        end
    end
    function obj = fit(obj,X,Y)
        [obj.C,~,obj.Y] = unique(Y);
        obj.X = X;
    end
    function Ypred = predict(obj,Xnew)
        P = size(Xnew,1);
        Ypred = zeros(P,1);
        for i = 1:P
            % Euclidean distance between two points
            A = repmat(Xnew(i,:),size(obj.X,1),1);
            distances = sqrt(sum((A-obj.X).^obj.p_norm,2));
            % Sort the distances in ascending order and check the k nearest training labels
            [~,I] = sort(distances);
            Ynearest = obj.Y(I(1:obj.k));
            % Frequencies of the k nearest training labels
            N = histc(Ynearest,1:max(Ynearest));
            frequencies = N(Ynearest);
            % Nearest training label with maximum frequency (if duplicated, check the nearest training instance)
            [~,J] = max(frequencies);
            Ypred(i) = Ynearest(J);
        end

        Ypred = obj.C(Ypred);
    end
    function [Xnearest,Ynearest,distances] = find(obj,Xnew)
        % Euclidean distance between two points
        A = repmat(Xnew,size(obj.X,1),1);
        distances = sqrt(sum((A-obj.X).^obj.p_norm,2));
        % Sort the distances in ascending order and check the k nearest training instances
        [distances,I] = sort(distances);
        Xnearest = obj.X(I(1:obj.k),:);
        Ynearest = obj.C(obj.Y(I(1:obj.k)));
        distances = distances(1:obj.k);
    end
end
end
