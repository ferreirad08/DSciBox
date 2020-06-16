classdef InformationGain
    properties
        k
        gain
        indexes
    end
    methods
        function obj = InformationGain(k)
            if nargin > 0
                obj.k = k;
            end
        end
        function obj = fit(obj,X,Y)
            [n_samples,n_features] = size(X);
            obj.gain = ones(1,n_features)*entropy(Y);
            for i = 1:n_features
                [~,~,feature] = unique(X(:,i));
                for j = 1:max(feature)
                    p = histc(feature(feature==j),j)/n_samples;
                    obj.gain(i) = obj.gain(i) - p*entropy(Y(feature==j));
                end
            end

            [obj.gain,obj.indexes] = sort(obj.gain,'descend');
        end
        function Xt = transform(obj,X)
            Xt = X(:,obj.indexes(1:obj.k));
        end
    end
end
