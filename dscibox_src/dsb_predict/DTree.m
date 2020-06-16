classdef DTree
    properties
        n
        b
        X
        C
        Y
        binranges
    end
    methods
        function obj = DTree(n)
            if nargin > 0
                obj.n = n;
            end
        end
        function obj = fit(obj,X,Y)
            % Discretize, if variables are continuous
            if isnumeric(X)
                % Calculate the number of bins based on Rice's Rule
                n_bins = ceil(2*numel(Y)^(1/3));
                obj.b = Binning(n_bins);
                obj.b = obj.b.fit(X);
                obj.X = obj.b.transform(X);
            else
                obj.X = X;
            end

            [obj.C,~,obj.Y] = unique(Y);
            obj.binranges = unique(obj.Y);
        end
        function label = predict(obj,Xnew)
            if isnumeric(Xnew)
                Xnew = obj.b.transform(Xnew);
            end
            
            P = size(Xnew,1);
            label = zeros(P,1);
            for i = 1:P
                Xcurrent = obj.X;
                Ycurrent = obj.Y;
                Xnewcurrent = Xnew(i,:);
                majority = [];
                while 1
                    frequencies = histc(Ycurrent,obj.binranges);
                    S = sum(frequencies);
                    M = max(frequencies);
                    I = find(frequencies==M);
                    % If there is a single class (pure node), the class will be selected
                    if S==M && M>0, label(i) = I; break, end
                    % Checks the majority class
                    if numel(I)==1, majority = I; end
                    % If there are no more features and the class has not been defined,
                    % the majority class will be selected
                    if S==0, label(i) = majority; break, end
                    % Branches the non-pure node
                    [Xcurrent,Ycurrent,Xnewcurrent] = branch(Xcurrent,Ycurrent,Xnewcurrent);
                end
            end

            label = obj.C(label);
        end
    end
end

function [X,Y,Xnew] = branch(X,Y,Xnew)
% Check the feature with the greatest information gain
[Xt,indexes,~] = feature_selection_gain(X,Y,1);
[str,~,values] = unique(Xt);
[~,value] = ismember(Xnew(indexes(1)),str);

X = X(values==value,:);
X(:,indexes(1)) = [];
Y = Y(values==value);
Xnew(indexes(1)) = [];
end
