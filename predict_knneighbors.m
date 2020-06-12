function label = predict_knneighbors(X,Y,Xnew,k)
%Author: David Ferreira - Federal University of Amazonas
%PhD student in Electrical Engineering
%Contact: ferreirad08@gmail.com
%
%k-Nearest Neighbors (kNN)
%
%Syntax
%1. label = predict_knneighbors(X,Y,Xnew,k)
%
%Description 
%1. Returns the estimated labels of one or multiple test instances.
%
%X is a M-by-N matrix, with M instances of N features. 
%Y is a M-by-1 matrix, with respective M labels to each training instance. 
%Xnew is a P-by-N matrix, with P instances of N features to be classified.
%k is a scalar, with the number of nearest neighbors selected.
%
%Examples
%1.
%     load fisheriris
%     X = meas;
%     Y = species;
%     Xnew = [min(meas);mean(meas);max(meas)];
%     k = 5;
%     label = predict_knneighbors(X,Y,Xnew,k)
%     label = 
%         'setosa'
%         'versicolor'
%         'virginica'

[C,~,Y] = unique(Y);

P = size(Xnew,1);
label = zeros(P,1);
for i = 1:P
    % Euclidean distance between two points
    A = repmat(Xnew(i,:),size(X,1),1);
    distances = sqrt(sum((A-X).^2,2));
    % Sort the distances in ascending order and check the k nearest training labels
    [~,I] = sort(distances);
    Ynearest = Y(I(1:k));
    % Frequencies of the k nearest training labels
    N = histc(Ynearest,1:max(Ynearest));
    frequencies = N(Ynearest);
    % Nearest training label with maximum frequency (if duplicated, check the nearest training instance)
    [~,J] = max(frequencies);
    label(i) = Ynearest(J);
end

label = C(label);
end
