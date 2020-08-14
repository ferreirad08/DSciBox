function accuracy = accuracy_score(Ynew,Ypred)
%Accuracy Classification Score
%
% SYNTAX
% 1. accuracy = dsb_utils.accuracy_score(Ynew,Ypred)
%
% DESCRIPTION
% 1. Returns the accuracy of the estimates (between 0 and 1).
%
% Ynew is a vector with real labels to each test instance.
% Ypred is a vector with estimated labels to each test instance.
%
% EXAMPLE
% 1.
%      Ynew = {'setosa';'versicolor';'virginica'};
%      Ypred = {'versicolor';'versicolor';'virginica'};
%      accuracy = dsb_utils.accuracy_score(Ynew,Ypred)
%      accuracy =
%          0.6667
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

if isrow(Ypred), Ypred = Ypred'; end

[C,~,Ynew] = unique(Ynew);
[~,Ypred] = ismember(Ypred,C);
accuracy = sum(Ynew==Ypred)/numel(Ynew);
end
