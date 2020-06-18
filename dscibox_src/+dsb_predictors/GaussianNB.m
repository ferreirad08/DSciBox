classdef GaussianNB
%Gaussian Naive Bayes (GNB)
%
% SYNTAX
% 1. mdl = dsb_predictors.GaussianNB()
%    mdl = mdl.fit(X,Y)
%    Ypred = mdl.predict(Xnew)
% 2. [Ysorted,probabilities] = mdl.find(Xnew(1,:))
%
% DESCRIPTION
% 1. Returns the estimated labels of one or multiple test instances.
% 2. Returns the probabilities of each label in relation to a new instance.
%
% X is a M-by-N matrix, with M instances of N features. 
% Y is a M-by-1 matrix, with respective M labels to each training instance. 
% Xnew is a P-by-N matrix, with P instances of N features to be classified.
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    n
    C
    n_class
    M
    S
    prior
end
methods
    function obj = GaussianNB(n)
        if nargin > 0
            obj.n = n;
        end
    end
    function obj = fit(obj,X,Y)
        [obj.C,~,Y] = unique(Y);
        obj.n_class = numel(obj.C);

        % Calculate the means and standard deviations
        obj.M = zeros(obj.n_class,size(X,2)); obj.S = obj.M;
        for i = 1:obj.n_class
            A = X(Y==i,:);
            obj.M(i,:) = mean(A);
            obj.S(i,:) = std(A,1);
        end

        % Class prior probability
        obj.prior = histc(Y,1:obj.n_class)/numel(Y);
    end
    function Ypred = predict(obj,Xnew)
        P = size(Xnew,1);
        Ypred = zeros(P,1);
        for i = 1:P
            % Repeats measurements in a matrix
            meas = repmat(Xnew(i,:),obj.n_class,1);
            % Probability density function (PDF) of the normal distribution
            gauss = 1./(obj.S.*sqrt(2.*pi))...
                .*exp(-1/2.*((meas-obj.M)./obj.S).^2);
            % Product
            probability = prod([gauss obj.prior],2);
            % Check the highest probability and the respective label
            [~,Ypred(i)] = max(probability);
        end

        Ypred = obj.C(Ypred);
    end
    function [Ysorted,probabilities] = find(obj,Xnew)
        % Repeats measurements in a matrix
        meas = repmat(Xnew,obj.n_class,1);
        % Probability density function (PDF) of the normal distribution
        gauss = 1./(obj.S.*sqrt(2.*pi))...
            .*exp(-1/2.*((meas-obj.M)./obj.S).^2);
        % Product
        probability = prod([gauss obj.prior],2);
        % Sort the normalized probabilities in descending order with their respective labels
        [probabilities,I] = sort(probability/sum(probability),'descend');
        Ysorted = obj.C(I);
    end
end
end
