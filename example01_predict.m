clear
clc

% Add library folders to the search path
addpath('dscibox_src')
% savepath

%%

fprintf('Classification examples using the Iris dataset with 30 percent for testing.\n');

load('datasets/fisheriris.mat')
[X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(meas,species,0.30,'stratified');

k = 5; % k must be an integer, 5 is the default
mdl = dsb_predictors.kNNeighbors(k,'euclidean');
mdl = mdl.fit(X,Y);
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred);
fprintf('Accuracy of %s: %d.\n','k-Nearest Neighbors',accuracy);

PDF = 'gaussian'; % 'gaussian' and 'exponential' are the options
mdl = dsb_predictors.NaiveBayes(PDF);
mdl = mdl.fit(X,Y);
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred);
fprintf('Accuracy of %s: %d.\n','Naive Bayes',accuracy);

mdl = dsb_predictors.DTree();
mdl = mdl.fit(X,Y);
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred);
fprintf('Accuracy of %s: %d.\n','Decision Tree',accuracy);

%%

fprintf('\nClassification examples using the Golf dataset with 25 percent for testing.\n');

load('datasets/golf-dataset.mat')
[X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(predictors,target,0.25,'stratified');

mdl = dsb_predictors.DTree();
mdl = mdl.fit(X,Y);
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred);
fprintf('Accuracy of %s: %d.\n','Decision Tree',accuracy);
