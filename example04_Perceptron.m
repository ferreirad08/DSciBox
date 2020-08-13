clear, clc
addpath('C:\Program Files\MATLAB\R2016a\toolbox\dscibox_src')
load fisheriris

[X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(meas(1:100,1:4),species(1:100),0.30,'stratified');
mdl = dsb_classification.Perceptron();
mdl = mdl.fit(X,Y)
Ypred = mdl.predict(Xnew);
accuracy = dsb_utilities.accuracy_score(Ynew,Ypred)
