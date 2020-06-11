function [label,model] = predict_gaussiannb(arg1,arg2,arg3)
%Author: David Ferreira - Federal University of Amazonas
%PhD student in Electrical Engineering
%Contact: ferreirad08@gmail.com
%
%Gaussian Naive Bayes (GNB)
%
%Syntax
%1. [label,model] = predict_gaussiannb(X,Y,Xnew)
%2. label = predict_gaussiannb(model,Xnew)
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
%     Xnew = [min(meas);max(meas)];
%     [label,model] = predict_gaussiannb(X,Y,Xnew)
%     label =
%         'setosa'
%         'virginica'
%     model = 
%              C               M               S           prior 
%         ____________    ____________    ____________    _______ 
%         'setosa'        [1x4 double]    [1x4 double]    0.33333
%         'versicolor'    [1x4 double]    [1x4 double]    0.33333
%         'virginica'     [1x4 double]    [1x4 double]    0.33333
%
%2.
%     Xnew = mean(meas);
%     label = predict_gaussiannb(model,Xnew)
%     label = 
%         'versicolor'

% Check the number of input
switch nargin
    case 2
        model = arg1;
        Xnew = arg2;
        n_class = numel(model.C);
    case 3
        X = arg1; 
        Y = arg2;
        Xnew = arg3;    
        [C,~,Y] = unique(Y);
        n_class = numel(C);

        % Calculate the means and standard deviations
        M = zeros(n_class,size(X,2)); S = M;
        for i = 1:n_class
            A = X(Y==i,:);
            M(i,:) = mean(A);
            S(i,:) = std(A,1);
        end

        % Class prior probability
        prior = histc(Y,1:n_class)/numel(Y);
        % Classification model
        model = table(C,M,S,prior);
    otherwise
        disp('Number of input arguments must be 2 or 3.')
end

n_tests = size(Xnew,1);
label = zeros(n_tests,1);
for i = 1:n_tests
    % Repeats measurements in a matrix
    meas = repmat(Xnew(i,:),n_class,1);
    % Probability density function (PDF) of the normal distribution
    gauss = 1./(model.S.*sqrt(2.*pi)).*exp(-1/2.*((meas-model.M)./model.S).^2);
    % Product
    probability = prod([gauss model.prior],2);
    % Check the highest probability and the respective label
    [~,label(i)] = max(probability);
end

label = model.C(label);
end
