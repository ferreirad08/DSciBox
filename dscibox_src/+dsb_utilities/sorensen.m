function D = sorensen(X,Y)
n = size(X,1);
k = size(Y,1);
D = zeros(n,k);
for i = 1:k
    A = repmat(Y(i,:),n,1);
    D(:,i) = sum(abs(A-X),2) ./ sum(A+X,2);
end
end
