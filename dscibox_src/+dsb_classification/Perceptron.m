classdef Perceptron
%Perceptron with Delta Rule

properties
    alpha = 0.01 % Learning Rate
    w
    bias = 0
end
methods
    function obj = Perceptron(alpha,n_features)
        if nargin > 0
            obj.alpha = alpha;
        end
        
        obj.w = 1-2.*rand(1,n_features);
    end
    function obj = fit(obj,X,Y)
        for i = 1:size(X,1)
            output = sum(X(i,:).*obj.w) + obj.bias;
            Ypred = output >= 0;
            e = Y(i) - Ypred;
            if e ~= 0
                obj.w = obj.w + obj.alpha*e.*X(i,:);
            end
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
