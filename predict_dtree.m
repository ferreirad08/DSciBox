function label = predict_dtree(X,Y,Xnew)
if isnumeric(X) && isnumeric(Xnew)
    Xt = binning([X; Xnew],3);
    n = size(X,1);
    X = Xt(1:n,:);
    Xnew = Xt(n+1:end,:);
end

while 1
    g = gain(X,Y);
    [~,I] = max(g);
    [str,~,values] = unique(X(:,I));
    [~,value] = ismember(Xnew(I),str);

    Y = Y(values==value);
    C = unique(Y);
    if numel(C)==1
        label = C;
        break
    end

    X = X(values==value,:); X(:,I) = [];
    Xnew(I) = [];
end
end

function g = gain(X,Y)
[n_samples,n_features] = size(X);
[~,~,Y] = unique(Y);
g = ones(1,n_features)*entropy(Y);
for i = 1:n_features
    feature = X(:,i);
    [~,~,feature] = unique(feature);
    for j=1:max(feature)
        p = histc(feature(feature==j),j)/n_samples;
        h = entropy(Y(feature==j));
        g(i) = g(i) - p*h;
    end
end
end
