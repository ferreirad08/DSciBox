function label = predict_dtree(X,Y,Xnew)
if isnumeric(X) && isnumeric(Xnew)
    n = size(X,1);
    n_bins = round(n^(1/3)*2); % Regra de Rice
    Xt = binning([X; Xnew],n_bins);
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

    X = X(values==value,:);
    X(:,I) = [];
    Xnew(I) = [];
end
end

function g = gain(X,Y)
[n_samples,n_features] = size(X);
[~,~,Y] = unique(Y);
g = ones(1,n_features)*entropy(Y);
for i = 1:n_features
    [~,~,feature] = unique(X(:,i));
    for j = 1:max(feature)
        p = histc(feature(feature==j),j)/n_samples;
        g(i) = g(i) - p*entropy(Y(feature==j));
    end
end
end
