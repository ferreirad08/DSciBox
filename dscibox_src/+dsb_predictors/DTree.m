classdef DTree
%Decision Tree (DT)
%
% SYNTAX
% 1. label = dsb_predictors.DTree(X,Y,Xnew)
%
% DESCRIPTION
% 1. Returns the estimated labels of one or multiple test instances.
%
% X is a M-by-N matrix, with M instances of N features. 
% Y is a M-by-1 matrix, with respective M labels to each training instance. 
% Xnew is a P-by-N matrix, with P instances of N features to be classified.
%
% EXAMPLE
% 1.
%      load fisheriris
%      X = meas;
%      Y = species;
%      Xnew = [min(meas);mean(meas);max(meas)];
%      label = dsb_predictors.DTree(X,Y,Xnew)
%      label = 
%          'setosa'
%          'versicolor'
%          'virginica'
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    n
    b
    X
    C
    Y
    binranges
end
methods
    function obj = DTree(n)
        if nargin > 0
            obj.n = n;
        end
    end
    function obj = fit(obj,X,Y)
        % Discretize, if variables are continuous
        if isnumeric(X)
            % Calculate the number of bins based on Rice's Rule
            n_bins = ceil(2*numel(Y)^(1/3));
            obj.b = dsb_preprocessing.Binning(n_bins);
            obj.b = obj.b.fit(X);
            obj.X = obj.b.transform(X);
        else
            obj.X = X;
        end

        [obj.C,~,obj.Y] = unique(Y);
        obj.binranges = unique(obj.Y);
    end
    function Ypred = predict(obj,Xnew)
        if isnumeric(Xnew)
            Xnew = obj.b.transform(Xnew);
        end

        P = size(Xnew,1);
        Ypred = zeros(P,1);
        for i = 1:P
            Xcurrent = obj.X;
            Ycurrent = obj.Y;
            Xnewcurrent = Xnew(i,:);
            majority = [];
            while 1
                frequencies = histc(Ycurrent,obj.binranges);
                S = sum(frequencies);
                M = max(frequencies);
                I = find(frequencies==M);
                % If there is a single class (pure node), the class will be selected
                if S==M && M>0, Ypred(i) = I; break, end
                % Checks the majority class
                if numel(I)==1, majority = I; end
                % If there are no more features and the class has not been defined,
                % the majority class will be selected
                if S==0, Ypred(i) = majority; break, end
                % Branches the non-pure node
                [Xcurrent,Ycurrent,Xnewcurrent] = branch(Xcurrent,Ycurrent,Xnewcurrent);
            end
        end

        Ypred = obj.C(Ypred);
    end
end
end

function [X,Y,Xnew] = branch(X,Y,Xnew)
% Check the feature with the greatest information gain
ig = dsb_preprocessing.InformationGain(1);
ig = ig.fit(X,Y);
Xr = ig.feature_selection(X);

[str,~,values] = unique(Xr);
[~,value] = ismember(Xnew(ig.indexes(1)),str);

X = X(values==value,:);
X(:,ig.indexes(1)) = [];
Y = Y(values==value);
Xnew(ig.indexes(1)) = [];
end
