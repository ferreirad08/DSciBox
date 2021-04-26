classdef MultivariateLinearRegression
%Multivariate Linear Regression
%
% SYNTAX
% 1. reg = dsb_regression.MultivariateLinearRegression()
%    reg = reg.fit(X, Y)
%    Ypred = reg.predict(Xnew)
%
% DESCRIPTION
% 1. Returns the regression to new observations of N explanatory (independent) variable. 
%
% X is a matrix with M observations of N explanatory (independent) variables.
% Y is a column vector with respective M observations of an explained (dependent)
% variable.
%
% EXAMPLE
% 1.
%      X = [1, 1; 1, 2; 2, 2; 2, 3]
%      y = [6; 8; 9; 11]
%      reg = dsb_regression.MultivariateLinearRegression().fit(X, y)
%      reg.coef_
%      reg.intercept_
%      reg.predict([3, 5])
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% D.Sc. student in Electrical Engineering from the Federal University of
% Amazonas 
% e-mail: ferreirad08@gmail.com

properties
    coef_
    intercept_
end
methods
    function obj = fit(obj, X, y)
        % Pseudo-inverse method
        X = [ones(size(X, 1), 1) X];
        obj.coef_ = inv(X'*X)*X'*y;
        obj.intercept_ = obj.coef_(1);
        obj.coef_(1) = [];
    end
    function y = predict(obj, X)
        X = [ones(size(X, 1), 1) X];
        y = X*[obj.intercept_ ; obj.coef_];
    end
end
end
