classdef Perceptron
%Perceptron with Delta Rule

properties
    eta = 0.01 % Learning Rate
    n_epochs = 2000
    w
    bias = -1
    C
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
        
        obj.w = rand(1,size(X,2)+1);

        for j = 1:obj.n_epochs
            cum_error = 0;
            for i = 1:size(X,1)
                output = sum([obj.bias, X(i,:)].*obj.w);
                Ypred = output >= 0; % Loss Function
                if Ypred ~= Y(i)
                    obj.w = obj.w + obj.eta*(Y(i) - Ypred).*[obj.bias, X(i,:)];
                    cum_error = cum_error + 1;
                end
            end
            if cum_error == 0, break, end
        end
    end
    function Ypred = predict(obj,Xnew)
        output = sum([repmat(obj.bias,size(Xnew,1),1), Xnew].*repmat(obj.w,size(Xnew,1),1),2);
        Ypred = output >= 0; % Loss Function
        Ypred = obj.C(Ypred+1);
    end
end
end
