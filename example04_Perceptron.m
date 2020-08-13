clear, clc

% Add library folders to the search path
addpath('dscibox_src')
% savepath

load('datasets/fisheriris.mat')

[X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(meas(1:100,1:4),species(1:100),0.30,'stratified');
mdl = dsb_classification.Perceptron();
mdl = mdl.fit(X,Y)
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred)
