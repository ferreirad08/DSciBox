load('16x220samples.mat')
X = samples(:,1:8); Y = samples(:,9);

load('fisheriris.mat')
X = meas; Y = species;

[X,Xnew,Y,Ynew] = data_sampling(X,Y,0.25);

label = predict_knneighbors(X,Y,Xnew,5);
accuracy = accuracy_score(Ynew,label)

label = predict_gaussiannb(X,Y,Xnew);
accuracy = accuracy_score(Ynew,label)

load('golf-dataset.mat')
X = predictors; Y = target;

[X,Xnew,Y,Ynew] = data_sampling(X,Y,0.05);

label = predict_dtree(X,Y,Xnew);
accuracy = accuracy_score(Ynew,label)
