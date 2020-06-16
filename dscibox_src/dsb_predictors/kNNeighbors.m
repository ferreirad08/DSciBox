classdef kNNeighbors
    properties
        k = 5
        C
        X
        Y
    end
    methods
        function obj = kNNeighbors(k)
            if nargin > 0
                obj.k = k;
            end
        end
        function obj = fit(obj,X,Y)
            [obj.C,~,obj.Y] = unique(Y);
            obj.X = X;
        end
        function label = predict(obj,Xnew)
            P = size(Xnew,1);
            label = zeros(P,1);
            for i = 1:P
                % Euclidean distance between two points
                A = repmat(Xnew(i,:),size(obj.X,1),1);
                distances = sqrt(sum((A-obj.X).^2,2));
                % Sort the distances in ascending order and check the k nearest training labels
                [~,I] = sort(distances);
                Ynearest = obj.Y(I(1:obj.k));
                % Frequencies of the k nearest training labels
                N = histc(Ynearest,1:max(Ynearest));
                frequencies = N(Ynearest);
                % Nearest training label with maximum frequency (if duplicated, check the nearest training instance)
                [~,J] = max(frequencies);
                label(i) = Ynearest(J);
            end

            label = obj.C(label);
        end
        function [Xnearest,Ynearest,distances] = find(obj,Xnew)
            % Euclidean distance between two points
            A = repmat(Xnew,size(obj.X,1),1);
            distances = sqrt(sum((A-obj.X).^2,2));
            % Sort the distances in ascending order and check the k nearest training instances
            [distances,I] = sort(distances);
            Xnearest = obj.X(I(1:obj.k),:);
            Ynearest = obj.C(obj.Y(I(1:obj.k)));
            distances = distances(1:obj.k);
        end
    end
end
