function [C2,i] = kMeans(X,arg2)
if nargin < 2
    arg2 = 8;
end

n_samples = size(X,1);
if isscalar(arg2)
    n_centroids = arg2;
    C = X(randperm(n_samples,n_centroids),:);
else
    C = arg2;
    n_centroids = size(C,1);
end

while 1
    [C2,i] = kMeans2(X,C,n_samples,n_centroids)
    if C == C2
        break
    end
    C = C2;
end
end

function [C,i] = kMeans2(X,C,n_samples,n_centroids)
distances = zeros(n_samples,n_centroids);
for i = 1:n_centroids
    A = repmat(C(i,:),n_samples,1) - X;
    distances(:,i) = vecnorm(A,2,2);
end

[~,i] = sort(distances,2);
i = i(:,1);

for j = 1:n_centroids
    C(j,:) = mean(X(i == j,:));
end
end
