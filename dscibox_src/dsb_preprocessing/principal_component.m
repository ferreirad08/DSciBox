function [Xt,coeff] = principal_component(X,arg2)
%Author: David Ferreira - Federal University of Amazonas
%PhD student in Electrical Engineering
%Contact: ferreirad08@gmail.com
%
%Principal Component Analysis
%
%Syntax
%1. [Xt,coeff] = principal_component(X,n_components)
%2. Xt = principal_component(X,coeff)
%
%Description
%1. Apply the PCA transformation in the features and returns the principal
%component coefficients. 
%2. Apply the PCA transformation in the features based on input
%coefficients. 
%
%X is a M-by-N matrix with features (continuous variables) in each column.
%n_components is the number of principal component.
%
%Examples
%1.
%     X = [1 2 3; 4 5 6; 7 8 9];
%     n_components = 2;
%     [Xt,coeff] = principal_component(X,n_components)
%     Xt =
%        -5.1962   -0.0000
%              0         0
%         5.1962    0.0000
%     coeff =
%         0.5774   -0.0332
%         0.5774   -0.6899
%         0.5774    0.7231
%
%2.
%     Xt = principal_component(X,coeff)
%     Xt =
%        -5.1962   -0.0000
%              0         0
%         5.1962    0.0000

% Calculates the mean of each column
M = mean(X);
% Centers the columns by subtracting column means
Xcentered = X - repmat(M,size(X,1),1);

if isscalar(arg2)
    n_components = arg2;
    % Calculates the covariance matrix of centered matrix
    V = cov(Xcentered);
    % Eigendecomposition of covariance matrix
    [vectors,values] = eig(V);
    % Sorts the eigenvalues ​​and associated eigenvectors
    [~,i] = sort(sum(values),'descend');
    % Project data
    coeff = vectors(:,i(1:n_components));
else
    coeff = arg2;
end

Xt = (coeff'*Xcentered')';
end
