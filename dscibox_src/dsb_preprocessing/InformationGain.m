classdef InformationGain
    properties
        n_features
        gain
        indexes
    end
    methods
        function obj = InformationGain(n_features)
            if nargin > 0
                obj.n_features = n_features;
            end
        end
        function obj = fit(obj,X,Y)
            [m,n] = size(X);
            obj.gain = ones(1,n)*entropy(Y);
            for i = 1:n
                [~,~,feature] = unique(X(:,i));
                for j = 1:max(feature)
                    p = histc(feature(feature==j),j)/m;
                    obj.gain(i) = obj.gain(i) - p*entropy(Y(feature==j));
                end
            end

            [obj.gain,obj.indexes] = sort(obj.gain,'descend');
        end
        function Xt = feature_selection(obj,X)
            Xt = X(:,obj.indexes(1:obj.n_features));
        end
    end
end
