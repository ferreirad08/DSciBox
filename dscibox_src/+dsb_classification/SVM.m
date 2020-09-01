classdef SVM
%Support Vector Machine
%
% David Alan de Oliveira Ferreira (http://lattes.cnpq.br/3863655668683045)
% D.Sc. student in Electrical Engineering from the Federal University of Amazonas
% e-mail: ferreirad08@gmail.com

properties
    C = 1
    eta = 0.001 % Learning Rate
    n_epochs = 500
    b = 0
    w % Synaptic Weights
end
methods
    function obj = SVM(C,eta,n_epochs)
        if nargin > 0
            obj.C = C;
        end
        if nargin > 1
            obj.eta = eta;
        end
        if nargin > 2
            obj.n_epochs = n_epochs;
        end
    end
    function obj = fit(obj,X,Y)
        %obj.w = [-0.24750747 1.50033755];
        obj.w = rand(1,size(X,2)); % Synaptic Weights

        margin__ = @(X,Y,w,b) Y .* (X * w' + b)

        for j = 1:obj.n_epochs
            margin = margin__(X,Y,obj.w,obj.b);
            idx = find(margin < 1);
            d_w = obj.w - obj.C * (Y(idx)'*X(idx,:));
            obj.w = obj.w - obj.eta * d_w;
            d_b = - obj.C * sum(Y(idx));
            obj.b = obj.b - obj.eta * d_b;
        end
    end
    function Ypred = predict(obj,Xnew)
        Ypred = Xnew * obj.w' + obj.b;
        Ypred(Ypred < 0) = -1;
        Ypred(Ypred > 0) = 1;
    end
end
end
