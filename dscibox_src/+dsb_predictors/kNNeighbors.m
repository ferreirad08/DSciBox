classdef kNNeighbors
%k-Nearest Neighbors (kNN)
%
% SYNTAX
% 1. mdl = dsb_predictors.kNNeighbors(k,p)
%    mdl = mdl.fit(X,Y)
%    Ypred = mdl.predict(Xnew)
% 2. [Xnearest,Ynearest,distances] = mdl.find(Xnew(1,:))
%
% DESCRIPTION
% 1. Returns the estimated labels of one or multiple test instances.
% 2. Returns the values of the features, labels and distances of the k
% nearest instances to a new instance. 
%
% X is a M-by-N matrix, with M instances of N features. 
% Y is a M-by-1 matrix, with respective M labels to each training instance. 
% Xnew is a P-by-N matrix, with P instances of N features to be classified.
% k is a scalar with the number of nearest neighbors selected.
% p is the power parameter for the distance metric.
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of
% Amazonas 
% e-mail: ferreirad08@gmail.com

properties
    k = 5
    p = 2
    C
    X
    Y
end
methods
    function obj = kNNeighbors(k,p)
        if nargin > 0
            obj.k = k;
        end
        if nargin > 1
            obj.p = p;
        end
    end
    function obj = fit(obj,X,Y)
        [obj.C,~,obj.Y] = unique(Y);
        obj.X = X;
    end
    function Ypred = predict(obj,Xnew)
        indices = find(obj,Xnew);
        Ynearest = obj.Y(indices)';
        % Frequencies of the k nearest training labels
        N = histc(Ynearest,1:max(obj.Y),1);
        [n_class,P] = size(N);
        frequencies = N(Ynearest...
            + repmat(0:n_class:P*n_class-n_class,obj.k,1));
        % Nearest training label with maximum frequency (if duplicated,
        % check the nearest training instance) 
        [~,J] = max(frequencies);
        Ypred = Ynearest(J + (0:obj.k:P*obj.k-obj.k));
        Ypred = obj.C(Ypred);
    end
    function [indices,distances] = find(obj,Xnew)
        % Distance between two points
        distances = dsb_utilities.cdist(Xnew,obj.X,obj.p);
        % Sort the distances in ascending order and check the k nearest
        % training labels 
        [distances,indices] = sort(distances,2);
        distances = distances(:,1:obj.k);
        indices = indices(:,1:obj.k);
    end
end
end
