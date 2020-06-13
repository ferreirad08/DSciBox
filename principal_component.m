function P = principal_component(X)
%Principal Component Analysis

% Calculates the mean of each column
M = mean(X);
% Centers the columns by subtracting column means
Xcentered = X - repmat(M,size(X,1),1);
% Calculates the covariance matrix of centered matrix
V = cov(Xcentered);
% Eigendecomposition of covariance matrix
[vectors,values] = eig(V);
% Sort eigenvalues ​​and associated eigenvectors
[~,i] = sort(sum(values),'descend');
% Project data
P = (vectors(:,i)'*Xcentered')';
end
