function [X,Xnew,Y,Ynew] = data_sampling(X,Y,p)
    % Simple or stratified random sampling.
    n = numel(Y);
    i = randperm(n,round(n*p));
    Xnew = X(i,:);
    X(i,:) = [];
    Ynew = Y(i);
    Y(i) = [];
end
