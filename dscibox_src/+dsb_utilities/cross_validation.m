function accuracy = cross_validation(mdl,X,Y,k)
%Cross Validation
%
% EXAMPLES
% 1.
%     load('fisheriris.mat')
%     mdl = dsb_predictors.kNNeighbors(5,2);
%     mdl = dsb_predictors.NaiveBayes('gaussian');
%     mdl = dsb_predictors.DTree();
%     k = 10;
%     accuracy = cross_validation(mdl,meas,species,k);
%     mu = mean(accuracy)
%     mu =
%         0.9467

if nargin < 4
    k = 10;
end
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
