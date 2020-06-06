function [label,accuracy] = predict_gnb(X,Y,Xnew,Ynew)
%Gaussian Naive Bayes (GNB)
%
%Author: David Ferreira - Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%predict_gnb
%
%Syntax
%1. label = predict_gnb(X,Y,Xnew)
%2. [label,accuracy] = predict_gnb(X,Y,Xnew,Ynew)
%
%Description 
%1. Returns the estimated labels of one or multiple test instances.
%2. Returns the estimated labels of one or multiple test instances and the accuracy of the estimates.
%
%X is a M-by-N matrix, with M instances of N features. 
%Y is a M-by-1 matrix, with respective M labels to each training instance. 
%Xnew is a P-by-N matrix, with P instances of N features to be classified.
%Ynew is a P-by-1 matrix, with respective P labels to each test instance (Used to check the accuracy between 0 and 1).
%
%Examples
%1.
%     load fisheriris
%     X = meas;
%     Y = species;
%     Xnew = [min(meas);mean(meas);max(meas)];
%     label = predict_gnb(X,Y,Xnew)
%     label = 
%         'setosa'
%         'versicolor'
%         'virginica'
%
%2.
%     Ynew = {'versicolor';'versicolor';'virginica'};
%     [label,accuracy] = predict_gnb(X,Y,Xnew,Ynew)
%     label = 
%         'setosa'
%         'versicolor'
%         'virginica'
%     accuracy =
%         0.6667

[C,~,Y] = unique(Y,'stable');
binranges = 1:numel(C);

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
    [~,I] = sort(probability,'descend');
    label(i) = I(1);
end

% Check the number of input and output arguments
if nargin > 3 && nargout > 1
    [~,Ynew] = ismember(Ynew,C);
    accuracy = sum(label==Ynew)/P;
end

label = C(label);
end
