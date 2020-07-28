clear
clc

% Add library folders to the search path
addpath('dscibox_src')
% savepath

%%

fprintf('Examples of clustering using the Iris dataset.\n');

load('datasets/fisheriris.mat')

k = 3; % Number of clusters
X = meas(:,3:4);
mdl = dsb_descriptors.kMeans(k).fit(X);

Xnew = [min(X);mean(X);max(X)]
Ypred = mdl.predict(Xnew)

hold on
plot(X(1:50,1),X(1:50,2),'o')
plot(X(51:100,1),X(51:100,2),'o')
plot(X(101:150,1),X(101:150,2),'o')
plot(mdl.C(:,1),mdl.C(:,2),'xk')
plot(Xnew(:,1),Xnew(:,2),'s')
hold off

legend({'setosa','versicolor','virginica','centroids','new instances'},'Location','northwest')
