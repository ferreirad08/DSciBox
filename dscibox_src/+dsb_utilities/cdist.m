function D = cdist(XA,XB,metric)
% coords = [35.0456, -85.2672;
%           35.1174, -89.9711;
%           35.9728, -83.9422;
%           36.1667, -86.7833];
% D = dsb_utilities.cdist(coords,coords,'euclidean')
% D =
%          0    4.7044    1.6172    1.8856
%     4.7044         0    6.0893    3.3561
%     1.6172    6.0893         0    2.8477
%     1.8856    3.3561    2.8477         0

if nargin < 3 || strcmp(metric,'euclidean')
    D = minkowski(XA,XB,2);
elseif strcmp(metric,'manhattan') || strcmp(metric,'cityblock')
    D = minkowski(XA,XB,1);
elseif strcmp(metric,'sorensen')
    D = sorensen(XA,XB);    
end
end

function D = minkowski(XA,XB,p)
n = size(XA,1);
k = size(XB,1);
D = zeros(n,k);
for i = 1:k
    A = repmat(XB(i,:),n,1);
    D(:,i) = dsb_utilities.vecnorm(A-XA,p,2);
end
end

function D = sorensen(XA,XB)
n = size(XA,1);
k = size(XB,1);
D = zeros(n,k);
for i = 1:k
    A = repmat(XB(i,:),n,1);
    D(:,i) = sum(abs(A-XA),2) ./ sum(A+XA,2);
end
end
