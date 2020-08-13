classdef Perceptron
%Perceptron with Delta Rule

properties
    n = 0.5 % learning rate
    bias = 0
    w
end
methods
    function obj = Perceptron(n,bias)
        if nargin > 0
            obj.n = n;
        end
        if nargin > 1
            obj.bias = bias;
        end
    end
    function obj = fit(obj,X,Y)
        obj.w = randn(1,size(X,2));
        for i = 1:size(X,1)
            output = sum(X(i,:).*obj.w) + obj.bias;
            Ypred = output >= 0;
            e = Y(i) - Ypred;
            if e ~= 0
                obj.w = obj.w + obj.n*e.*X(i,:);
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
