classdef LinearRegression
%Linear Regression
%
% SYNTAX
% 1. reg = dsb_predictors.LinearRegression()
%    reg = reg.fit(X,Y)
%    [Ypred,R_squared] = reg.predict(Xnew)
%
% DESCRIPTION
% 1. Returns the estimated labels of one or multiple test instances.
% 2. Returns the values of the features, labels and distances of the k
% nearest instances to a new instance. 
%
% X is a vector with M observations of an explanatory (independent) variable.
% Y is a vector with respective M observations of an explained (dependent)
% variable.
% R_squared is the Pearson correlation coefficient (rho) squared.
%
% EXAMPLE
% 1.
%      X = [2,3,9,6];
%      Y = [5,7,12,8];
%      reg = LinearRegression().fit(X,Y);
%      Ynew = [2,3,9,6];
%      [Ypred,R_squared] = reg.predict(Xnew)
%      Ypred =
%          5.3000    6.2000   11.6000    8.9000
%      R_squared =
%          0.9346
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% PhD student in Electrical Engineering from the Federal University of
% Amazonas 
% e-mail: ferreirad08@gmail.com

properties
    n
    coeff
    Y
end
methods
    function obj = LinearRegression(n)
        if nargin > 0
            obj.n = n;
        end
    end
    function obj = fit(obj,X,Y)
        % calculating the linear regression coefficients
        a = cov(X,Y)/var(X);
        b = mean(Y)-a*mean(X);
        obj.coeff = [a, b];
        obj.Y = Y;
    end
    function [Ypred,R_squared] = predict(obj,Xnew)
        % estimating points of the line
        Ypred = obj.coeff(1)*Xnew+obj.coeff(2);
        
        % calculating the Pearson correlation coefficient (rho) and the
        % determination coefficient (R^2) 
        rho = cov(obj.Y,Ypred)/(std(obj.Y)*std(Ypred));
        R_squared = rho*rho;
    end
end
end

% defining the covariance function
function r = cov(x,y)
    r = mean((x-mean(x)).*(y-mean(y)));
end

% defining the variance function
function r = var(x)
    r = cov(x,x);
end

% defining the standard deviation function
function r = std(x)
    r = var(x)^(1/2);
end
