clear
clc

X = [1 2;
     3 4;
     5 6];
 
n_components = 2;

pca = PCA(n_components);
pca = pca.fit(X);
Xt = pca.transform(X)
