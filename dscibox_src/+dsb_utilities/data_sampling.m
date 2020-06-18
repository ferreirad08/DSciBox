function [X,Xnew,Y,Ynew] = data_sampling(X,Y,p,type)
%Simple or Stratified Random Sampling
%
% SYNTAX
% 1. [X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(X,Y,p) % 'simple' is the default
% 2. [X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(X,Y,p,'stratified')
%
% DESCRIPTION
% 1. Splits data regardless of labels.
% 2. Splits data proportionally to labels.
%
% X is a M-by-N matrix, with M instances of N features. 
% Y is a M-by-1 matrix, with respective M labels to each training instance. 
% p is a float number between 0.0 and 1.0 and represent the proportion of the dataset to include in the test set.
%
% EXAMPLE
% 1.
%      load fisheriris
%      p = 0.30;
%      [X,Xnew,Y,Ynew] = dsb_utilities.data_sampling(meas,species,p,'stratified')
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

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
