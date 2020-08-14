classdef Perceptron
%Perceptron with Delta Rule

properties
    eta = 0.01 % Learning Rate
    n_epochs = 2000
    bias = -1
    w % Synaptic Weights
    C % Class Names
end
methods
    function obj = Perceptron(eta,n_epochs,bias)
        if nargin > 0
            obj.eta = eta;
        end
        if nargin > 1
            obj.n_epochs = n_epochs;
        end
        if nargin > 2
            obj.bias = bias;
        end
    end
    function obj = fit(obj,X,Y)
        [obj.C,~,Y] = unique(Y); Y = Y-1;
        [n,m] = size(X);
        X = [repmat(obj.bias,n,1), X];
        obj.w = rand(1,m+1);

        for j = 1:obj.n_epochs
            cum_error = 0;
            for i = 1:n
                u = sum(X(i,:).*obj.w); % Activation Potential
                Ypred = u >= 0; % Activation Function
                if Ypred ~= Y(i)
                    obj.w = obj.w + obj.eta*(Y(i) - Ypred).*X(i,:);
                    cum_error = cum_error + 1;
                end
            end
            if cum_error == 0, break, end
        end
    end
    function Ypred = predict(obj,Xnew)
        n = size(Xnew,1);
        Xnew = [repmat(obj.bias,n,1), Xnew];
        u = sum(Xnew.*repmat(obj.w,n,1),2); % Activation Potential
        Ypred = u >= 0; % Activation Function
        Ypred = obj.C(Ypred+1);
    end
end
end
