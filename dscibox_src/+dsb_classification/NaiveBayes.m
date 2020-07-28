classdef NaiveBayes
%Naive Bayes (NB)
%
% SYNTAX
% 1. mdl = dsb_predictors.NaiveBayes(PDF) % 'gaussian' and 'exponential' are the options
%    mdl = mdl.fit(X,Y)
%    Ypred = mdl.predict(Xnew)
% 2. [Ysorted,probabilities] = mdl.find(Xnew(1,:))
%
% DESCRIPTION
% 1. Returns the estimated labels of one or multiple test instances.
% 2. Returns the probabilities of each label in relation to a new instance.
%
% PDF is the distribution of the numerical variables (features).
% X is a M-by-N matrix, with M instances of N features.
% Y is a M-by-1 matrix, with respective M labels to each training instance. 
% Xnew is a P-by-N matrix, with P instances of N features to be classified.
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    PDF = 'gaussian'
    C
    n_class
    mu
    sigma
    prior
end
methods
    function obj = NaiveBayes(PDF)
        if nargin > 0
            obj.PDF = PDF;
        end
    end
    function obj = fit(obj,X,Y)
        [obj.C,~,Y] = unique(Y);
        obj.n_class = numel(obj.C);

        % Calculate the means and standard deviations
        obj.mu = zeros(obj.n_class,size(X,2)); obj.sigma = obj.mu;
        for i = 1:obj.n_class
            A = X(Y==i,:);
            obj.mu(i,:) = mean(A);
            obj.sigma(i,:) = std(A,1);
        end

        % Class prior probability
        obj.prior = histc(Y,1:obj.n_class)/numel(Y);
    end
    function Ypred = predict(obj,Xnew)
        P = size(Xnew,1);
        Ypred = zeros(P,1);
        for i = 1:P
            % Class posterior probability
            [~,~,I] = find(obj,Xnew(i,:));
            % Check the label with highest probability
            Ypred(i) = I(1);
        end

        Ypred = obj.C(Ypred);
    end
    function [Ysorted,probabilities,I] = find(obj,Xnew)
        % Repeats measurements in a matrix
        meas = repmat(Xnew,obj.n_class,1);
        if strcmp(obj.PDF,'gaussian')
            % Probability density function (PDF) of the normal distribution
            p = dsb_utilities.normpdf(meas,obj.mu,obj.sigma);
        elseif strcmp(obj.PDF,'exponential')
            % PDF of the exponential distribution
            p = dsb_utilities.exppdf(meas,obj.mu);
        end
        % Product
        probability = prod([p obj.prior],2);
        % Sort the normalized probabilities in descending order
        [probabilities,I] = sort(probability/sum(probability),'descend');
        % Sort the labels in descending order
        Ysorted = obj.C(I);
    end
end
end
