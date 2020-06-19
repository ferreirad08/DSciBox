clear
clc

% Add library folders to the search path
addpath('dscibox_src')
% savepath

%%

fprintf('Examples using the Iris dataset.\n');

load('fisheriris.mat')
Xnew = mean(meas); % Creating a new instance from the mean values

fprintf('\nIn the k-Nearest Neighbors, it is possible to obtain the values of the features,\n labels and distances of the k nearest instances to a new instance.\n');

k = 5; % k must be an integer, 5 is the default
p = 2; % p must be an integer, 2 is the default
mdl = dsb_predictors.kNNeighbors(k,p);
mdl = mdl.fit(meas,species);
[Xnearest,Ynearest,distances] = mdl.find(Xnew)

fprintf('\nIn Naive Bayes, it is possible to obtain the probabilities of each label\n in relation to a new instance.\n');

PDF = 'gaussian'; % 'gaussian' and 'exponential' are the options
mdl = dsb_predictors.NaiveBayes(PDF);
mdl = mdl.fit(meas,species);
[Ysorted,probabilities] = mdl.find(Xnew)
