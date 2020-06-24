classdef kMeans
%k-Means (kM)
%
% SYNTAX
% 1. mdl = dsb_predictors.kMeans(arg1) % arg1 is the number of clusters or the centroids of each cluster
%    mdl = mdl.fit(X)
%    Ypred = mdl.predict(Xnew)
%
% DESCRIPTION
% 1. Returns the estimated clusters of one or multiple test instances.
%
% k is a scalar with the number of clusters selected.
% X is a M-by-N matrix, with M instances of N features.
% Xnew is a P-by-N matrix, with P instances of N features for clustering.
%
% EXAMPLES
% 1.
% >> X = [[1, 2]; [1, 4]; [1, 0];[10, 2]; [10, 4]; [10, 0]];
% >> k = 2;
% >> mdl = dsb_descriptors.kMeans(k).fit(X);
% >> Xnew = [[0, 0]; [12, 3]];
% >> Ypred = mdl.predict(Xnew)
% 
% Ypred =
% 
%      1
%      2
% 
% >> mdl.C % Centroids
% 
% ans =
% 
%      1     2
%     10     2
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    k = 2
    C = []
    idx
end
methods
    function obj = kMeans(arg1)
        if nargin > 0
            if isscalar(arg1)
                obj.k = arg1;
            else
                obj.C = arg1;
                obj.k = size(arg1,1);
            end
        end
    end
    function obj = fit(obj,X)
        n_samples = size(X,1);
        if isempty(obj.C)
            obj.C = X(randperm(n_samples,obj.k),:);
        end
        
        while 1
            [obj.idx,Cnew] = ordinary_function(X,obj.C,n_samples,obj.k);
            if obj.C == Cnew
                break
            end
            obj.C = Cnew;
        end
    end
    function Ypred = predict(obj,Xnew)
        P = size(Xnew,1);
        Ypred = zeros(P,1);
        for i = 1:P
            A = repmat(Xnew(i,:),obj.k,1) - obj.C;
            distances = dsb_utilities.vecnorm(A,2,2);
            [~,J] = sort(distances);
            Ypred(i) = J(1);
        end
    end
end
end

function [idx,C] = ordinary_function(X,C,n_samples,k)
distances = zeros(n_samples,k);
for idx = 1:k
    A = repmat(C(idx,:),n_samples,1) - X;
    distances(:,idx) = dsb_utilities.vecnorm(A,2,2);
end

[~,idx] = sort(distances,2);
idx = idx(:,1);
for j = 1:k
    C(j,:) = mean(X(idx == j,:));
end
end
