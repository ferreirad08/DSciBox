# DSciBox (Data Science Toolbox)

[![View Gaussian Naive Bayes (GNB) Classifier on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/76355-gaussian-naive-bayes-gnb-classifier)

Author: [David Ferreira](http://lattes.cnpq.br/3863655668683045)

% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

*Currently available functions:*

Preprocessing
        
    Pricipal Component Analysis [OK]
        pca = PCA(n_components)
        pca = pca.fit(X)
        Xt = pca.transform(X)
    
    Quantile Binning Transformation [OK]
        b = dsb_preprocessing.Binning(n_bins)
        b = b.fit(X)
        Xt = b.transform(X)
    
    Feature Selection Based on Information Gain [OK]
        ig = dsb_preprocessing.InformationGain(n_features)
        ig = ig.fit(X,Y)
        Xr = ig.feature_selection(X)
    
    Feature Selection Based on Chi-squared
        cs = ChiSquared(n_features)
        cs = cs.fit(X,Y)
        Xr = cs.feature_selection(X)

Utilities [OK]

    Simple or Stratified Random Sampling
        [X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(X,Y,0.30,'stratified') % 'simple' is the default
    
    Accuracy Classification Score
        accuracy = dsb_utilities.accuracy_score(Ynew,Ypred)

    Information Entropy
        e = dsb_utilities.entropy(Y)

    Quantile Analysis
        Q = dsb_utilities.quantile(X,[0.25 0.50 0.75])
        
Predictors

    k-Nearest Neighbors [OK]
        mdl = dsb_predictors.kNNeighbors(k)
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)
        [Xnearest,Ynearest,distances] = mdl.find(Xnew(1,:))

    Gaussian Naive Bayes [OK]
        mdl = dsb_predictors.GaussianNB()
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)
        [Ysorted,probabilities] = mdl.find(Xnew(1,:))

    Decision Tree [OK]
        mdl = dsb_predictors.DTree()
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)

    Random Forest
        mdl = RandomForest(n_trees)
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)

    Suport Vector Machine
        mdl = SVM()
        mdl = mdl.fit(X,Y)
        Ypred = mdl.predict(Xnew)

Descriptors

    Artificial Neural Network
        mdl = ANN()
        result = mdl.descript()
