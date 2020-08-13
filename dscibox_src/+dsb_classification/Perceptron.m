classdef Perceptron
%Perceptron with Delta Rule

properties
    alpha = 0.01 % Learning Rate
    n_iter = 2000
    w
    bias = 0
end
methods
    function obj = Perceptron(alpha,n_features,n_iter)
        obj.alpha = alpha;
        obj.n_iter = n_iter;
        obj.w = 1-2.*rand(1,n_features);
    end
    function obj = fit(obj,X,Y)
        for j = 1:obj.n_iter
            cum_error = [];
            for i = 1:size(X,1)
                output = sum(X(i,:).*obj.w) + obj.bias;
                Ypred = output >= 0;
                error = Y(i) - Ypred;
                cum_error = [cum_error,error];
                if error ~= 0
                    obj.w = obj.w + obj.alpha*error.*X(i,:);
                end
            end
            if cum_error == 0, break, end
        end
    end
    function Ypred = predict(obj,Xnew)
        P = size(Xnew,1);
        Ypred = zeros(P,1);
        for i = 1:P
            output = sum(Xnew(i,:).*obj.w) + obj.bias;
            Ypred(i) = output >= 0;
        end
    end
end
end
