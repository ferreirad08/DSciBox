function [idx,C] = kMeans(X,arg2)
if nargin < 2
    arg2 = 8;
end

n_samples = size(X,1);
if isscalar(arg2)
    k = arg2;
    C = X(randperm(n_samples,k),:);
else
    C = arg2;
    k = size(C,1);
end

while 1
    [idx,Cnew] = ordinary_function(X,C,n_samples,k);
    if C == Cnew
        break
    end
    C = Cnew;
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
