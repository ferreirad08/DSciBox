# DSciBox (Data Science Toolbox)

[![View DSciBox on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/77067-dscibox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ferreirad08/DSciBox/blob/master/LICENSE)
[![Linkedin](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/david-f-3a918ba5)

Dscibox: Practical Data Science Tools and Techniques with MATLAB Implementations

### Preprocessing
        
Pricipal Component Analysis
```matlab
pca = dsb_preprocessing.PCA(n_components)
pca = pca.fit(X)
Xt = pca.transform(X)
```
Min-Max Normalizer
```matlab
scaler = dsb_preprocessing.MinMaxNormalizer()
scaler = scaler.fit(X)
Xt = scaler.transform(X)
```
Quantile Binning Transformation
```matlab
b = dsb_preprocessing.Binning(n_bins)
b = b.fit(X)
Xt = b.transform(X)
```
Feature Selection Based on Information Gain
```matlab
ig = dsb_preprocessing.InformationGain(n_features)
ig = ig.fit(X,Y)
Xr = ig.feature_selection(X)
```

### Utilities

Simple or Stratified Random Sampling
```matlab
        [X,Xnew,Y,Ynew] = dsb_utils.data_sampling(X,Y,0.30,'stratified')
```
Cross Validation
        accuracy = dsb_utils.cross_validation(mdl,X,Y,k)

Accuracy Classification Score
        accuracy = dsb_utils.accuracy_score(Ynew,Ypred)

Information Entropy
        e = dsb_utils.entropy(Y)

Quantile Analysis
        Q = dsb_utils.quantile(X,[0.25 0.50 0.75])

Distance Matrix
        D = dsb_utils.cdist(XA,XB,'euclidean')
        
### Classification

k-Nearest Neighbors
        mdl = dsb_classification.kNNeighbors(k,'euclidean')
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)
        [indices,distances] = mdl.find(Xnew)

Naive Bayes
        mdl = dsb_classification.NaiveBayes('gaussian')
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)
        [Ysorted,probabilities] = mdl.find(Xnew(1,:))

Decision Tree
        mdl = dsb_classification.DTree()
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)

Perceptron Network
        mdl = dsb_classification.Perceptron(eta,n_epochs,bias)
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)

### Regression

Simple Linear Regression
        reg = dsb_regression.SimpleLinearRegression()
        reg = reg.fit(X,Y)
        Ypred = reg.predict(Xnew)

Multivariate Linear Regression
        reg = dsb_regression.MultivariateLinearRegression()
        reg = reg.fit(X,Y)
        Ypred = reg.predict(Xnew)

k-Nearest Neighbors Regressor
        mdl = dsb_regression.kNNeighborsRegressor(k,'euclidean',weights);
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)
        
### Clustering

k-Means
        mdl = dsb_clustering.kMeans(k)
        mdl = mdl.fit(X)
        Ypred = mdl.predict(Xnew)
