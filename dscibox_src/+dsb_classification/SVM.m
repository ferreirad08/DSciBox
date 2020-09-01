load('fisheriris.mat')
[X,Xnew,Y,Ynew] = data_sampling(meas(1:100,[1 3]),species(1:100),0.10,'stratified')
y_train = ones(100,1);
y_train(1:50) = -1;

train_f1 = X(:,1)';
train_f2 = X(:,2)';

w1 = zeros(size(train_f1));
w2 = zeros(size(train_f2));

epochs = 1;
alpha = 0.0001;

while(epochs < 10000)
    y = w1 .* train_f1 + w2 .* train_f2;
    prod = y .* y_train;
    epochs
    count = 1;
    for val = prod
        if(val >= 1)
            cost = 0;
            w1 = w1 - alpha * (2 * 1/epochs * w1);
            w2 = w2 - alpha * (2 * 1/epochs * w2);
            
        else
            cost = 1 - val;
            w1 = w1 + alpha * (train_f1(count) * y_train(count) - 2 * 1/epochs * w1);
            w2 = w2 + alpha * (train_f2(count) * y_train(count) - 2 * 1/epochs * w2);
        count = count + 1;
    epochs = epochs + 1;
        end
    end
end
