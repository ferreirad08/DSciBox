function label = predict_dtree(X,Y,Xnew)
%Author: David Ferreira - Federal University of Amazonas
%PhD student in Electrical Engineering
%Contact: ferreirad08@gmail.com
%
%Decision Tree (DT)
%
%Syntax
%1. label = predict_dtree(X,Y,Xnew)
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
%     Xnew = [min(meas);mean(meas);max(meas)];
%     label = predict_dtree(X,Y,Xnew)
%     label = 
%         'setosa'
%         'versicolor'
%         'virginica'

% Discretize, if variables are continuous
if isnumeric(X) && isnumeric(Xnew)
    % Calculate the number of bins based on Rice's Rule
    n_bins = ceil(2*numel(Y)^(1/3));
    [X,Q] = binning(X,n_bins);
    Xnew = binning(Xnew,Q);
end

[C,~,Y] = unique(Y);
binranges = unique(Y);

P = size(Xnew,1);
label = zeros(P,1);
for i = 1:P
    Xcurrent = X;
    Ycurrent = Y;
    Xnewcurrent = Xnew(i,:);
    majority = [];
    while 1
        frequencies = histc(Ycurrent,binranges);
        S = sum(frequencies);
        M = max(frequencies);
        I = find(frequencies==M);
        % If there is a single class (pure node), the class will be selected
        if S==M && M>0, label(i) = I; break, end
        % Check the majority class
        if numel(I)==1, majority = I; end
        % If there are no more features and the class has not been defined,
        % the majority class will be selected
        if S==0, label(i) = majority; break, end
        % Branches the non-pure node
        [Xcurrent,Ycurrent,Xnewcurrent] = branch(Xcurrent,Ycurrent,Xnewcurrent);
    end
end

label = C(label);
end

function [X,Y,Xnew] = branch(X,Y,Xnew)
% Check the feature with the greatest gain of information
[~,I] = max(gain(X,Y));
[str,~,values] = unique(X(:,I));
[~,value] = ismember(Xnew(I),str);

X = X(values==value,:);
X(:,I) = [];
Y = Y(values==value);
Xnew(I) = [];
end

function g = gain(X,Y)
[n_samples,n_features] = size(X);
g = ones(1,n_features)*entropy(Y);
for i = 1:n_features
    [~,~,feature] = unique(X(:,i));
    for j = 1:max(feature)
        p = histc(feature(feature==j),j)/n_samples;
        g(i) = g(i) - p*entropy(Y(feature==j));
    end
end
end
