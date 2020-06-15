function [labels,probabilities] = find_gaussiannb(X,Y,Xnew)
%Author: David Ferreira
%PhD student in Electrical Engineering
%Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%Gaussian Naive Bayes (GNB)
%
%Syntax
%1. [labels,probabilities] = find_gaussiannb(X,Y,Xnew)
%
%Description
%1. Returns the labels with their respective probabilities in descending order.
%
%X is a M-by-N matrix, with M instances of N features. 
%Y is a M-by-1 matrix, with respective M labels to each training instance. 
%Xnew is a 1-by-N matrix, with one instance of N features to be classified.
%
%Examples
%1.
%     load fisheriris
%     X = meas;
%     Y = species;
%     Xnew = mean(meas);
%     [labels,probabilities] = find_gaussiannb(X,Y,Xnew)
%     labels = 
%         'versicolor'
%         'virginica'
%         'setosa'
%     probabilities =
%         1.0000
%         0.0000
%         0.0000

[C,~,Y] = unique(Y);
n_class = numel(C);

% Calculate the means and standard deviations
M = zeros(n_class,size(X,2)); S = M;
for i = 1:n_class
    A = X(Y==i,:);
    M(i,:) = mean(A);
    S(i,:) = std(A,1);
end

% Class prior probability
prior = histc(Y,1:n_class)/numel(Y);
% Repeats measurements in a matrix
meas = repmat(Xnew,n_class,1);
% Probability density function (PDF) of the normal distribution
gauss = 1./(S.*sqrt(2.*pi)).*exp(-1/2.*((meas-M)./S).^2);
% Product
probability = prod([gauss prior],2);
% Sort the normalized probabilities in descending order with their respective labels
[probabilities,I] = sort(probability/sum(probability),'descend');
labels = C(I);
end
