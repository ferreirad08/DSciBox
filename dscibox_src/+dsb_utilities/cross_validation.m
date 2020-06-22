function accuracy = cross_validation(mdl,X,Y,k)
n = numel(Y);
if n < k
    disp('n < k')
end
n = n-rem(n,k);
i = reshape(randperm(n),n/k,k);

accuracy = zeros(1,k);
for fold = 1:k
    l = i(:,fold);

    Xtrain = X;
    Ytrain = Y;
    
    Xnew = Xtrain(l,:);
    Xtrain(l,:) = [];
    Ynew = Ytrain(l);
    Ytrain(l) = [];

    mdl = mdl.fit(Xtrain,Ytrain);
    Ypred = mdl.predict(Xnew);
    accuracy(fold) = dsb_utilities.accuracy_score(Ynew,Ypred);
end
end
