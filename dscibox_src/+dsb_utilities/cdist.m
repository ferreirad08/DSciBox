function D = cdist(X,Y,p)
if nargin < 3
    p = 2; % p-norm
end
n = size(X,1);
k = size(Y,1);
D = zeros(n,k);
for i = 1:k
    A = repmat(Y(i,:),n,1) - X;
    D(:,i) = dsb_utilities.vecnorm(A,p,2);
end
end
