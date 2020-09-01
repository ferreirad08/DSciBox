load('fisheriris.mat')
[X,Xnew,Y,Ynew] = data_sampling(meas(1:100,[1 3]),species(1:100),0.10,'stratified')
plot(X(:,1),X(:,2),'o')

w = zeros(size(X))

epochs = 1
alpha = 0.0001
