classdef PCA
    properties
        n_components
        coeff
    end
    methods
        function obj = PCA(n_components)
            if nargin > 0
                obj.n_components = n_components;
            end
        end
        function obj = fit(obj,X)
            % Calculates the mean of each column
            M = mean(X);
            % Centers the columns by subtracting column means
            Xcentered = X - repmat(M,size(X,1),1);
            % Calculates the covariance matrix of centered matrix
            V = cov(Xcentered);
            % Eigendecomposition of covariance matrix
            [vectors,values] = eig(V);
            % Sorts the eigenvalues ??and associated eigenvectors
            [~,i] = sort(sum(values),'descend');
            % Project data
            obj.coeff = vectors(:,i(1:obj.n_components));
        end
        function Xt = transform(obj,X)
            % Calculates the mean of each column
            M = mean(X);
            % Centers the columns by subtracting column means
            Xcentered = X - repmat(M,size(X,1),1);
            Xt = (obj.coeff'*Xcentered')';
        end
    end
end
