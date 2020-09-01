close, clear, clc
addpath('C:\Program Files\MATLAB\R2016a\toolbox\dscibox_src')
load fisheriris

X = meas(51:150,[3 4])
y = species(51:150)
[~,~,y] = unique(y)
y(y == 1) = -1
y(y == 2) = 1

plot(X(1:50,1),X(1:50,2),'o',X(51:100,1),X(51:100,2),'s')

X = dsb_preprocessing.StandardData().fit(X).transform(X)

C = 15
eta = 1e-3
n_epochs = 500
[n,m] = size(X)
w = [-0.24750747  1.50033755]%rand(1,m)
b = 0

margin = @(X,y,w,b) y .* (X * w' + b)

for j = 1:n_epochs
    margin_v = margin(X,y,w,b);
    idx = find(margin_v < 1);
    d_w = w - C * (y(idx)'*X(idx,:));
    w = w - eta * d_w;
    d_b = - C * sum(y(idx));
    b = b - eta * d_b
end
margin_v = margin(X,y,w,b);
support_vectors = find(margin_v <= 1)

Ypred = X * w' + b;
Ypred(Ypred < 0) = -1;
Ypred(Ypred > 0) = 1;
accuracy = mean(y == Ypred)
