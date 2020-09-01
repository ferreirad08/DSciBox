clear, clc, load fisheriris
addpath('C:\Program Files\MATLAB\R2016a\toolbox\dscibox_src')
y_train(1:50) = -1;
y_train(51:100) = 1;
[X,Xnew,Y,Ynew] = dsb_utils.data_sampling(meas(1:100,[1 3]),y_train,0.10,'stratified');

eta = 0.0001; % Learning Rate
n_epochs = 10000; % Number of Epochs
[n,m] = size(X);
w = zeros(n,m); % Synaptic Weights

for epoch = 1:n_epochs
    lambda = 1/epoch; % Regularization Parameter
    
    y = sum(w.*X,2);
    prod = y .* Y;
    
    for i = 1:n
        if prod(i) >= 1
            cost = 0;
            w = w - eta * (2 * lambda * w);
        else
            cost = 1 - prod(i);
            w = w + eta * (repmat(y_train(i) * X(i,:),n,1) - 2 * lambda * w);
        end
    end
end
w
