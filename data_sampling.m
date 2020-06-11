function [X,Xnew,Y,Ynew] = data_sampling(X,Y,p)
%Author: David Ferreira - Federal University of Amazonas
%PhD student in Electrical Engineering
%Contact: ferreirad08@gmail.com
%
%Simple or Stratified Random Sampling
%
%Syntax
%1. [X,Xnew,Y,Ynew] = data_sampling(X,Y,p)
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
%     p = 0.25;
%     [X,Xnew,Y,Ynew] = data_sampling(X,Y,p)
%     label = 
%         'setosa'
%         'versicolor'
%         'virginica'

n = numel(Y);
i = randperm(n,round(n*p));
Xnew = X(i,:);
X(i,:) = [];
Ynew = Y(i);
Y(i) = [];
end
