function [X,Xnew,Y,Ynew] = data_sampling(X,Y,p,type)
%Author: David Ferreira - Federal University of Amazonas
%PhD student in Electrical Engineering
%Contact: ferreirad08@gmail.com
%
%Simple or Stratified Random Sampling
%
%Syntax
%1. [X,Xnew,Y,Ynew] = data_sampling(X,Y,p)
%1. [X,Xnew,Y,Ynew] = data_sampling(X,Y,p,'stratified')
%
%Description 
%1. Returns the estimated labels of one or multiple test instances.
%
%X is a M-by-N matrix, with M instances of N features. 
%Y is a M-by-1 matrix, with respective M labels to each training instance. 
%p is a float number between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
%
%Examples
%1.
%     load fisheriris
%     X = meas;
%     Y = species;
%     p = 0.25;
%     [X,Xnew,Y,Ynew] = data_sampling(X,Y,p)

if nargin > 3 && strcmp(type,'stratified')
    [~,~,Ynumerical] = unique(Y);
    i = [];
    for j = 1:max(Ynumerical)
        k = find(Ynumerical==j);
        n = numel(k);
        i = [i; k(randperm(n,round(n*p)))];
    end
else
    n = numel(Y);
    i = randperm(n,round(n*p));
end

Xnew = X(i,:);
X(i,:) = [];
Ynew = Y(i);
Y(i) = [];
end
