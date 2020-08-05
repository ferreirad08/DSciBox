# DSciBox (Data Science Toolbox)

[![View DSciBox on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/77067-dscibox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ferreirad08/DSciBox/blob/master/LICENSE)
[![in: LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/david-f-3a918ba5)

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 84 21" data-supported-dps="84x21" width="84" height="21" focusable="false">
  <g class="inbug" fill="none" fill-rule="evenodd">
    <path d="M82.042 0H64.146c-.856 0-1.583.677-1.583 1.511v17.977c0 .835.727 1.512 1.583 1.512h17.896c.857 0 1.52-.677 1.52-1.512V1.511C83.563.677 82.9 0 82.043 0" class="bug-text-color" fill="#FFF"></path>
    <path d="M82.042 0H64.146c-.856 0-1.583.677-1.583 1.511v17.977c0 .835.727 1.512 1.583 1.512h17.896c.857 0 1.52-.677 1.52-1.512V1.511C83.563.677 82.9 0 82.043 0zM70.875 7.875h2.844v1.429c.41-.822 1.46-1.56 3.038-1.56 3.025 0 3.743 1.635 3.743 4.636v5.557h-3.063v-4.874c0-1.709-.41-2.672-1.452-2.672-1.446 0-2.048 1.039-2.048 2.672v4.874h-3.062V7.876zm-5.25 10.063h3.063V7.874h-3.063v10.063zm3.5-13.344a1.969 1.969 0 11-3.938 0 1.969 1.969 0 013.938 0z" class="background" fill="#0073B0"></path>
  </g>
  <g class="linkedin-text">
    <path d="M59.5 17.938h-2.625v-1.532c-.478.727-1.87 1.663-3.281 1.663-2.99 0-4.944-1.892-4.944-4.944 0-2.803 1.641-5.381 4.506-5.381 1.287 0 2.576.25 3.282 1.225V3.063H59.5v14.874zm-5.578-7.482c-1.557 0-2.34.935-2.34 2.45 0 1.516.783 2.494 2.34 2.494s2.56-.978 2.56-2.494c0-1.515-1.003-2.45-2.56-2.45zM47.25 16.242c-.997 1.267-2.994 1.827-4.922 1.827-3.114 0-5.053-2.164-5.053-5.381 0-3.218 2.319-4.944 5.49-4.944 2.626 0 4.835 1.845 4.835 5.381 0 .547-.088.875-.088.875H40.25l.049.293c.174.964 1.208 1.457 2.412 1.457 1.017 0 1.778-.443 2.297-1.148l2.242 1.64zm-2.734-4.43c.02-1.038-.814-1.88-1.935-1.88-1.37 0-2.248.905-2.331 1.88h4.266z"></path>
    <path d="M27.563 3.063h3.062v8.75l3.5-3.938h3.828l-4.266 4.594 4.102 5.469h-3.664l-3.5-4.813v4.813h-3.063zM16.188 7.875h2.625v1.477c.414-.79 1.766-1.608 3.28-1.608 3.156 0 3.72 1.862 3.72 4.56v5.633H22.75v-4.976c0-1.162.067-2.548-1.531-2.548-1.619 0-1.969 1.24-1.969 2.548v4.976h-3.063V7.876zM12.906 2.68c1.012 0 1.914.903 1.914 1.914 0 1.01-.902 1.914-1.914 1.914-1.01 0-1.914-.903-1.914-1.914s.904-1.914 1.914-1.914zm-1.531 15.258h3.063V7.874h-3.063v10.063zM0 3.063h3.063v11.812h6.562v3.063H0z"></path>
  </g>
</svg>

*Currently available functions:*

Preprocessing
        
    Pricipal Component Analysis
        pca = dsb_preprocessing.PCA(n_components)
        pca = pca.fit(X)
        Xt = pca.transform(X)
    
    Min-Max Normalizer
        scaler = dsb_preprocessing.MinMaxNormalizer()
        scaler = scaler.fit(X)
        Xt = scaler.transform(X)
    
    Quantile Binning Transformation
        b = dsb_preprocessing.Binning(n_bins)
        b = b.fit(X)
        Xt = b.transform(X)
    
    Feature Selection Based on Information Gain
        ig = dsb_preprocessing.InformationGain(n_features)
        ig = ig.fit(X,Y)
        Xr = ig.feature_selection(X)

Utilities

    Simple or Stratified Random Sampling
        [X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(X,Y,0.30,'stratified')
    
    Cross Validation
        accuracy = cross_validation(mdl,X,Y,k)

    Accuracy Classification Score
        accuracy = dsb_utilities.accuracy_score(Ynew,Ypred)

    Information Entropy
        e = dsb_utilities.entropy(Y)

    Quantile Analysis
        Q = dsb_utilities.quantile(X,[0.25 0.50 0.75])
        
Classification

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

Regression

    Linear Regression
        reg = dsb_regression.LinearRegression()
        reg = reg.fit(X,Y)
        Ypred = reg.predict(Xnew)
        
Clustering

    k-Means
        mdl = dsb_clustering.kMeans(k)
        mdl = mdl.fit(X)
        Ypred = mdl.predict(Xnew)
