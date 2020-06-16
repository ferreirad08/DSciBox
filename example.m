clear
clc

X = [1 2;
     3 4;
     5 6];
 
n_components = 2;

pca = PCA(n_components);
pca = pca.fit(X);
Xt = pca.transform(X)

% ----------------------

X = [16     2;
      5    11;
      9     7;
      4    14];

n_bins = 3;

b = Binning(n_bins);
b = b.fit(X);
Xt = b.transform(X)

% ----------------------

load fisheriris
Xnew = [min(meas);mean(meas);max(meas)]

m = GaussianNB();
m = m.fit(meas,species);
label = m.predict(Xnew)
[labels,probabilities] = m.find(Xnew(2,:))

% ----------------------

k = 5
mk = kNNeighbors(k);
mk = mk.fit(meas,species);
label = mk.predict(Xnew)
[Xnearest,Ynearest,distances] = mk.find(Xnew(2,:))














