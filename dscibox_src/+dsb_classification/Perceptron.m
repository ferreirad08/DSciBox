classdef Perceptron
%Perceptron with Delta Rule

properties
    alpha = 0.01 % Learning Rate
    n_epochs = 2000
    w
    bias
end
methods
    function obj = Perceptron(alpha,n_epochs)
        if nargin > 0
            obj.alpha = alpha;
        end
        if nargin > 1
            obj.n_iter = n_iter;
        end
    end
    function obj = fit(obj,X,Y)
        obj.w = rand(1,size(X,2));
        obj.bias = rand;

        for j = 1:obj.n_epochs
            cum_error = 0;
            for i = 1:size(X,1)
                output = sum(X(i,:).*obj.w) + obj.bias;
                Ypred = output >= 0; % Loss Function
                error = Y(i) - Ypred;
                if error ~= 0
                    cum_error = cum_error + 1;
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
