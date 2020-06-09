function label = predict_dtree(X,Y,Xnew)
if isnumeric(X) && isnumeric(Xnew)
    n = size(X,1);
    n_bins = round(n^(1/3)*2); % Regra de Rice
    Xt = binning([X; Xnew],n_bins);
    X = Xt(1:n,:);
    Xnew = Xt(n+1:end,:);
end

[C,~,Y] = unique(Y);

if isnumeric(C)
    C(end+1) = NaN;
else
    C(end+1) = {'None'};
end

P = size(Xnew,1);
label = zeros(P,1);
for i = 1:P
    X_current = X;
    Y_current = Y;
    Xnew_current = Xnew(i,:);
    while 1
        label_current = unique(Y_current);
        if numel(label_current)==1, label(i) = label_current; break, end
        if numel(label_current)==0, label(i) = numel(C); break, end
        [X_current,Y_current,Xnew_current] = branch(X_current,Y_current,Xnew_current);
    end
end

label = C(label);
end

function [X,Yd,Xnew] = branch(X,Yd,Xnew)
g = gain(X,Yd);
[~,I] = max(g);
[str,~,values] = unique(X(:,I));
[~,value] = ismember(Xnew(I),str);

X = X(values==value,:); X(:,I) = [];
Yd = Yd(values==value);
Xnew(I) = [];
end

function g = gain(X,Ys)
[n_samples,n_features] = size(X);
[~,~,Ys] = unique(Ys);
g = ones(1,n_features)*entropy(Ys);
for i = 1:n_features
    [~,~,feature] = unique(X(:,i));
    for j = 1:max(feature)
        p = histc(feature(feature==j),j)/n_samples;
        g(i) = g(i) - p*entropy(Ys(feature==j));
    end
end
end
