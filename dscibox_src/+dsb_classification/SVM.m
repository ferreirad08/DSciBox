clear, clc, load fisheriris
addpath('C:\Program Files\MATLAB\R2016a\toolbox\dscibox_src')
y_train(1:50,1) = -1;
y_train(51:100,1) = 1;
[X,Xnew,Y,Ynew] = dsb_utils.data_sampling(meas(1:100,[1 3]),y_train,0.10,'stratified');

eta = 0.0001; % Learning Rate
n_epochs = 10000; % Number of Epochs
[n,m] = size(X);
w = zeros(1,m); % Synaptic Weights

for j = 1:n_epochs
    lambda = 1/j; % Regularization Parameter    
    for i = 1:n
        y = sum(w.*X(i,:));
        prod = Y(i) * y;
        if prod >= 1
            w = w - eta * (2 * lambda * w);
        else
            w = w + eta * (Y(i) * X(i,:) - 2 * lambda * w);
        end
    end
end
w
