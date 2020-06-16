classdef Binning
    properties
        n_bins
        Q
    end
    methods
        function obj = Binning(n_bins)
            if nargin > 0
                obj.n_bins = n_bins;
            end
        end
        function obj = fit(obj,X)
            p = (1:obj.n_bins-1)/obj.n_bins;
            obj.Q = quantile(X,p);
        end
        function Xt = discret(obj,X)
            [m,n] = size(X);
            Xt = zeros(m,n);
            for i = 1:n
                Xt(:,i) = sum(repmat(X(:,i)',obj.n_bins-1,1)...
                    >=repmat(obj.Q(:,i),1,m));
            end
        end
    end
end
