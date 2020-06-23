load('fisheriris.mat')

k = 3; % Number of clusters
p = 2; % p-norm
mdl = dsb_descriptors.kMeans(k,p);

X = meas(:,3:4);
mdl = mdl.fit(X);

Xnew = [min(X);mean(X);max(X)];
Ypred = mdl.predict(Xnew);

hold on
plot(X(1:50,1),X(1:50,2),'o')
plot(X(51:100,1),X(51:100,2),'o')
plot(X(101:150,1),X(101:150,2),'o')
plot(mdl.C(:,1),mdl.C(:,2),'xk')
plot(Xnew(:,1),Xnew(:,2),'s')
hold off

legend({'setosa','versicolor','virginica','centroids','new instances'},'Location','northwest')
