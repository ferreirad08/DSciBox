load('datasets/fisheriris.mat')

X = meas(51:150,[3 4])
X = dsb_prepocessing.StandardData().fit(X).transform(X)
Y(1:50) = -1
Y(51:100) = 1

C = 15
mdl = SVM(C)
mdl = mdl.fit(X,Y)
Ypred = mdl.predict(X)
accuracy = mean(Y == Ypred)
