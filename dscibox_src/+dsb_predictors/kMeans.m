classdef kMeans
%k-Means (kM)
%
% SYNTAX
% 1. mdl = dsb_predictors.kNNeighbors(k,p)
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
% p is the power parameter for the distance metric.
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    k = 2
    p = 2
    C = []
    idx
end
methods
    function obj = kMeans(arg1,p)
        if nargin > 0
            if isscalar(arg1)
                obj.k = arg1;
            else
                obj.C = arg1;
                obj.k = size(C,1);
            end
        end
        if nargin > 1
            obj.p = p;
        end
    end
    function obj = fit(obj,X)
        n_samples = size(X,1);
        if isempty(obj.C)
            obj.C = X(randperm(n_samples,obj.k),:);
        end
        
        while 1
            [obj.idx,Cnew] = ordinary_function(X,obj.C,n_samples,obj.k,obj.p);
            if obj.C == Cnew
                break
            end
            obj.C = Cnew;
        end
    end
end
end

function [idx,C] = ordinary_function(X,C,n_samples,k,p)
distances = zeros(n_samples,k);
for idx = 1:k
    A = repmat(C(idx,:),n_samples,1) - X;
    distances(:,idx) = dsb_utilities.vecnorm(A,p,2);
end

[~,idx] = sort(distances,2);
idx = idx(:,1);
for j = 1:k
    C(j,:) = mean(X(idx == j,:));
end
end
