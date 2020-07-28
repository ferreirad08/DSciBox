clear
clc

% Add library folders to the search path
addpath('dscibox_src')
% savepath

%%

fprintf('Examples using the Iris dataset.\n');

load('datasets/fisheriris.mat')
Xnew = mean(meas); % Creating a new instance from the mean values

fprintf('\nIn the k-Nearest Neighbors, it is possible to obtain the indices and\n distances of the k nearest instances to a new instance.\n');

k = 5; % k must be an integer, 5 is the default
mdl = dsb_classification.kNNeighbors(k,'euclidean');
mdl = mdl.fit(meas,species);
[indices,distances] = mdl.find([min(meas);mean(meas);max(meas)])

fprintf('\nIn Naive Bayes, it is possible to obtain the probabilities of each label\n in relation to a new instance.\n');

PDF = 'gaussian'; % 'gaussian' and 'exponential' are the options
mdl = dsb_classification.NaiveBayes(PDF);
mdl = mdl.fit(meas,species);
[Ysorted,probabilities] = mdl.find(Xnew)
