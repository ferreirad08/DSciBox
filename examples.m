clc

% load('golf-dataset.mat')
% [X,Xnew,Y,Ynew] = data_sampling(predictors,target,0.25);
% 
% load('16x220samples.mat')
% [X,Xnew,Y,Ynew] = data_sampling(samples(:,1:8),samples(:,9),0.25);

load('fisheriris.mat')
[X,Xnew,Y,Ynew] = data_sampling(meas,species,0.25);

label = predict_knneighbors(X,Y,Xnew,5);
accuracy = accuracy_score(Ynew,label)

label = predict_gaussiannb(X,Y,Xnew);
accuracy = accuracy_score(Ynew,label)

label(i) = predict_dtree(X,Y,Xnew);
accuracy = accuracy_score(Ynew,label)
