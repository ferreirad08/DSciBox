classdef PCA
%Principal Component Analysis
%
% SYNTAX
% 1. pca = PCA(n_components)
%    pca = pca.fit(X)
%    Xt = pca.transform(X)
%
% DESCRIPTION
% 1. Apply the PCA transformation in the features.
%
% X is a M-by-N matrix with features (continuous variables) in each column.
% n_components is the number of principal component (features) in Xt (transformed set).
%
% EXAMPLE
% 1.
%      X = [1 2 3; 4 5 6; 7 8 9];
%      n_components = 2;
%      pca = PCA(n_components)
%      pca = pca.fit(X)
%      Xt = pca.transform(X)
%      Xt =
%         -5.1962   -0.0000
%               0         0
%          5.1962    0.0000
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    n_components
    mu
    coeff
end
methods
    function obj = PCA(n_components)
        if nargin > 0
            obj.n_components = n_components;
        end
    end
    function obj = fit(obj,X)
        % Calculates the mean of each column
        obj.mu = mean(X);
        % Centers the columns by subtracting column means
        Xcentered = X - repmat(obj.mu,size(X,1),1);
        % Calculates the covariance matrix of centered matrix
        C = cov(Xcentered);
        % Eigendecomposition of covariance matrix
        [vectors,values] = eig(C);
        % Sorts the eigenvalues and associated eigenvectors
        [~,i] = sort(sum(values),'descend');
        % Selects the desired number of coefficients
        obj.coeff = vectors(:,i(1:obj.n_components));
    end
    function Xt = transform(obj,X)
        % Centers the columns by subtracting column means
        Xcentered = X - repmat(obj.mu,size(X,1),1);
        % Project data
        Xt = (obj.coeff'*Xcentered')';
    end
end
end
