classdef PCA
%Principal Component Analysis
%
% SYNTAX
% 1. [Xt,coeff] = principal_component(X,n_components)
% 2. Xt = principal_component(X,coeff)
%
% DESCRIPTION
% 1. Apply the PCA transformation in the features and returns the principal component coefficients. 
% 2. Apply the PCA transformation in the features based on input coefficients. 
%
% X is a M-by-N matrix with features (continuous variables) in each column.
% n_components is the number of principal component.
%
% EXAMPLE
% 1.
%      X = [1 2 3; 4 5 6; 7 8 9];
%      n_components = 2;
%      [Xt,coeff] = principal_component(X,n_components)
%      Xt =
%         -5.1962   -0.0000
%               0         0
%          5.1962    0.0000
%      coeff =
%          0.5774   -0.0332
%          0.5774   -0.6899
%          0.5774    0.7231
%
% 2.
%      Xt = principal_component(X,coeff)
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
        M = mean(X);
        % Centers the columns by subtracting column means
        Xcentered = X - repmat(M,size(X,1),1);
        % Calculates the covariance matrix of centered matrix
        V = cov(Xcentered);
        % Eigendecomposition of covariance matrix
        [vectors,values] = eig(V);
        % Sorts the eigenvalues and associated eigenvectors
        [~,i] = sort(sum(values),'descend');
        % Selects the desired number of coefficients
        obj.coeff = vectors(:,i(1:obj.n_components));
    end
    function Xt = transform(obj,X)
        % Calculates the mean of each column
        M = mean(X);
        % Centers the columns by subtracting column means
        Xcentered = X - repmat(M,size(X,1),1);
        % Project data
        Xt = (obj.coeff'*Xcentered')';
    end
end
end
