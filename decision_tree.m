function label = decision_tree(X,Y,Xnew)  
    while 1
        [X,Y,Xnew] = sub_set(X,Y,Xnew);
        C = unique(Y);
        if numel(C)==1
            label = C;
            break 
        end
    end
end

function [X,Y,Xnew] = sub_set(X,Y,Xnew)
    g = gain(X,Y);
    [~,feature] = max(g);
    [str,~,values] = unique(X(:,feature));
    [~,value] = ismember(Xnew(feature),str);
    X = X(values==value,:);
    X(:,feature) = [];
    Y = Y(values==value);
    Xnew(feature) = [];
end

function g = gain(X,Y)
    [n_samples,n_features] = size(X);
    [~,~,Y] = unique(Y);
    g = ones(1,n_features)*entropY(Y);
    for i = 1:n_features
        feature = X(:,i);
        [~,~,feature] = unique(feature);
        for j=1:max(feature)
            p = histc(feature(feature==j),j)/n_samples;
            e = entropY(Y(feature==j));
            g(i) = g(i) - p*e;
        end
    end
end

function e = entropY(Y)
    p = histc(Y,unique(Y))/numel(Y);
    e = sum(-p.*log2(p));
end
