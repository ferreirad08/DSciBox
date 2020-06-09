function label = predict_gaussiannb(X,Y,Xnew)
%Author: David Ferreira - Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%Gaussian Naive Bayes (GNB)
%
%Syntax
%1. label = predict_gaussiannb(X,Y,Xnew)
%
%Description 
%1. Returns the estimated labels of one or multiple test instances.
%
%X is a M-by-N matrix, with M instances of N features. 
%Y is a M-by-1 matrix, with respective M labels to each training instance. 
%Xnew is a P-by-N matrix, with P instances of N features to be classified.
%
%Examples
%1.
%     load fisheriris
%     X = meas;
%     Y = species;
%     Xnew = [min(meas);mean(meas);max(meas)];
%     label = predict_gaussiannb(X,Y,Xnew)
%     label = 
%         'setosa'
%         'versicolor'
%         'virginica'

[C,~,Y] = unique(Y);
binranges = unique(Y)';

P = size(Xnew,1);
label = zeros(P,1);
for i = 1:P
    % Class prior probability
    probability = histc(Y,binranges)/numel(Y);
    for j = binranges
        A = X(Y==j,:);
        S = std(A,1);
        % Probability density function (PDF) of the normal distribution
        gauss = 1./(S.*sqrt(2.*pi))...
            .*exp(-1/2.*((Xnew(i,:)-mean(A))./S).^2);
        % Product
        probability(j) = probability(j)*prod(gauss);
    end
    
    % Sort the probabilities in descending order and check the estimated label
    [~,label(i)] = max(probability);
end

label = C(label);
end
