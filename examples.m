clc

% Add library folders to the search path
addpath('datasets')
addpath('dscibox_src/dsb_find',...
        'dscibox_src/dsb_predict',...
        'dscibox_src/dsb_transformation',...
        'dscibox_src/dsb_utils')
% savepath

disp('Fisher iris')
load('fisheriris.mat')
[X,Xnew,Y,Ynew] = data_sampling(meas,species,0.25,'stratified');

label = predict_knneighbors(X,Y,Xnew,5);
disp(['kNN: ' num2str(accuracy_score(Ynew,label))])

label = predict_gaussiannb(X,Y,Xnew);
disp(['GNB: ' num2str(accuracy_score(Ynew,label))])

label = predict_dtree(X,Y,Xnew);
disp(['DT: ' num2str(accuracy_score(Ynew,label))])

%%

disp(' ')
disp('Golf dataset')
load('golf-dataset.mat')
[X,Xnew,Y,Ynew] = data_sampling(predictors,target,0.25,'stratified');

label = predict_dtree(X,Y,Xnew);
disp(['DT: ' num2str(accuracy_score(Ynew,label))])
