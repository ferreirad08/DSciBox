function accuracy = accuracy_score(Ynew,Ypred)
%Accuracy
%
%Author: David Ferreira - Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%Syntax
%1. accuracy = accuracy_score(Ynew,Ypred)
%
%Description 
%1. Returns the accuracy of the estimates (between 0 and 1).
%
%Ynew is a vector with real labels to each test instance.
%Ypred is a vector with estimated labels to each test instance.
%
%Examples
%1.
%     Ynew = {'setosa';'versicolor';'virginica'};
%     Ypred = {'versicolor';'versicolor';'virginica'};
%     accuracy = accuracy_score(Ynew,Ypred)
%     accuracy =
%         0.6667

if isrow(Ypred), Ypred = Ypred'; end

[C,~,Ynew] = unique(Ynew);
[~,Ypred] = ismember(Ypred,C);
accuracy = sum(Ynew==Ypred)/numel(Ynew);
end