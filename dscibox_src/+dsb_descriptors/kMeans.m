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
        % k-centroid initialization
        if isempty(obj.C)
            Xcopy = X;
            i = randi(size(Xcopy,1));
            obj.C(1,:) = Xcopy(i,:);
            Xcopy(i,:) = [];
            for j = 2:obj.k
                D = dsb_utilities.cdist(Xcopy,obj.C).^2;
                [~,i] = max(min(D,[],2));
                obj.C(j,:) = Xcopy(i,:);
                Xcopy(i,:) = [];
            end
        end
        
        % convergence
        Cnew = zeros(size(obj.C));
        while 1
            obj.idx = predict(obj,X);
            for j = 1:obj.k
                Cnew(j,:) = mean(X(obj.idx == j,:));
            end
            
            if obj.C == Cnew, break, end
            obj.C = Cnew;
        end
    end
    function Ypred = predict(obj,Xnew)
        D = dsb_utilities.cdist(Xnew,obj.C);
        [~,Ypred] = min(D,[],2);
    end
end
end
