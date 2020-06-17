clc

% Add library folders to the search path
addpath('dscibox_src')
% savepath

disp('Examples of classification using the Iris Dataset')

load('fisheriris.mat')
[X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(meas,species,0.30,'stratified');

k = 5; % k must be an integer, 5 is the default
mdl = dsb_predictors.kNNeighbors(k);
mdl = mdl.fit(X,Y);
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred);
disp(['kNN: ' num2str(accuracy)])
% [Xnearest,Ynearest,distances] = mdl.find(Xnew(2,:))

mdl = dsb_predictors.GaussianNB();
mdl = mdl.fit(X,Y);
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred);
disp(['GNB: ' num2str(accuracy)])
% [Ysorted,probabilities] = mdl.find(Xnew(2,:))

mdl = dsb_predictors.DTree();
mdl = mdl.fit(X,Y);
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred);
disp(['DT: ' num2str(accuracy)])

%%

disp(' ')
disp('Examples of classification using the Golf Dataset')

load('golf-dataset.mat')
[X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(predictors,target,0.30,'stratified');

mdl = dsb_predictors.DTree();
mdl = mdl.fit(X,Y);
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred);
disp(['DT: ' num2str(accuracy)])
