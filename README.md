# DSciBox (Data Science Toolbox)

[![View Gaussian Naive Bayes (GNB) Classifier on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/76355-gaussian-naive-bayes-gnb-classifier)

Author: [David Ferreira](http://lattes.cnpq.br/3863655668683045)

*Currently available functions:*

Preprocessing
        
    Pricipal Component Analysis [OK]
        pca = PCA(n_components)
        pca = pca.fit(X)
        Xt = pca.transform(X)
    
    Quantile Binning Transformation [OK]
        b = Binning(n_bins)
        b = b.fit(X)
        Xt = b.transform(X)
    
    Feature Selection Based on Information Gain [OK]
        ig = InformationGain(n_features)
        ig = ig.fit(X,Y)
        Xr = ig.feature_selection(X)
    
    Feature Selection Based on Chi-squared
        cs = ChiSquared(n_features)
        cs = cs.fit(X,Y)
        Xr = cs.feature_selection(X)

Utilities [OK]

    Simple or Stratified Random Sampling
        [X,Xnew,Y,Ynew] = data_sampling(X,Y,0.30,'stratified') % 'simple' is the default
    Accuracy Classification Score
        accuracy = accuracy_score(Ynew,Ypred)
    Information Entropy
        e = entropy(Y)
    Quantile Analysis
        Q = quantile(X,p)
        
Predictors

    kNNeighbors (k-Nearest Neighbors) [OK]
        fit
        predict
        find
    GaussianNB (Gaussian Naive Bayes) [OK]
        fit
        predict
        find
    DTree (Decision Tree) [OK]
        fit
        predict
    RandomForest (Random Forest)
        fit
        predict
        find
    SVM (Suport Vector Machine)
        fit
        predict

Descriptors

    ANN (Artificial Neural Network)
        descript
