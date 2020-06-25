function [Ypred,I] = kNN(X,Y,Xnew,k)
[C,~,Y] = unique(Y);
% calculate distances between instances
D = dsb_utilities.cdist(Xnew,X);
% sorts the distances in ascending order
[~,I] = sort(D,2);
% select the k nearest instances
I = I(:,1:k);
Ynearest = Y(I);
% count the frequencies of the labels
frequencies = histc(Ynearest,1:max(Y),2);
% select the most frequent label
[~,J] = max(frequencies,[],2);
Ypred = C(J);
end
