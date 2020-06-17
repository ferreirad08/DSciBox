clc

% Add library folders to the search path
addpath('datasets',...
        'dscibox_src')
% savepath

disp('Fisher iris')
load('fisheriris.mat')
[X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(meas,species,0.30,'stratified');

k = 5; % k must be an integer, 5 is the default
mk = dsb_predictors.kNNeighbors(k);
mk = mk.fit(X,Y);
label = mk.predict(Xnew);
disp(['kNN: ' num2str(dsb_utilities.accuracy_score(Ynew,label))])
% [Xnearest,Ynearest,distances] = mk.find(Xnew(2,:))

mg = dsb_predictors.GaussianNB();
mg = mg.fit(X,Y);
label = mg.predict(Xnew);
disp(['GNB: ' num2str(dsb_utilities.accuracy_score(Ynew,label))])
% [labels,probabilities] = mg.find(Xnew(2,:))

mdt = dsb_predictors.DTree();
mdt = mdt.fit(X,Y);
label = mdt.predict(Xnew);
disp(['DT: ' num2str(dsb_utilities.accuracy_score(Ynew,label))])

%%

disp(' ')
disp('Golf dataset')
load('golf-dataset.mat')
[X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(predictors,target,0.30,'stratified');

mdt2 = dsb_predictors.DTree();
mdt2 = mdt2.fit(X,Y);
label = mdt2.predict(Xnew);
disp(['DT: ' num2str(dsb_utilities.accuracy_score(Ynew,label))])
