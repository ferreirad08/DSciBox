classdef GaussianNB
    properties
        n_bins
        C
        n_class
        M
        S
        prior
    end
    methods
        function obj = GaussianNB(n_bins)
            if nargin > 0
                obj.n_bins = n_bins;
            end
        end
        function obj = fit(obj,X,Y)
            [obj.C,~,Y] = unique(Y);
            obj.n_class = numel(obj.C);

            % Calculate the means and standard deviations
            obj.M = zeros(obj.n_class,size(X,2)); obj.S = obj.M;
            for i = 1:obj.n_class
                A = X(Y==i,:);
                obj.M(i,:) = mean(A);
                obj.S(i,:) = std(A,1);
            end

            % Class prior probability
            obj.prior = histc(Y,1:obj.n_class)/numel(Y);
        end
        function Ypred = predict(obj,Xnew)
            P = size(Xnew,1);
            Ypred = zeros(P,1);
            for i = 1:P
                % Repeats measurements in a matrix
                meas = repmat(Xnew(i,:),obj.n_class,1);
                % Probability density function (PDF) of the normal distribution
                gauss = 1./(obj.S.*sqrt(2.*pi))...
                    .*exp(-1/2.*((meas-obj.M)./obj.S).^2);
                % Product
                probability = prod([gauss obj.prior],2);
                % Check the highest probability and the respective label
                [~,Ypred(i)] = max(probability);
            end

            Ypred = obj.C(Ypred);
        end
        function [Yunique,probabilities] = find(obj,Xnew)
            % Repeats measurements in a matrix
            meas = repmat(Xnew,obj.n_class,1);
            % Probability density function (PDF) of the normal distribution
            gauss = 1./(obj.S.*sqrt(2.*pi))...
                .*exp(-1/2.*((meas-obj.M)./obj.S).^2);
            % Product
            probability = prod([gauss obj.prior],2);
            % Sort the normalized probabilities in descending order with their respective labels
            [probabilities,I] = sort(probability/sum(probability),'descend');
            Yunique = obj.C(I);
        end
    end
end
