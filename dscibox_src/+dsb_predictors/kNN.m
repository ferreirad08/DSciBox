function [Ypred,I] = kNN(X,Y,Xnew,k)
[C,~,Y] = unique(Y);
D = dsb_utilities.cdist(Xnew,X);
[~,I] = sort(D,2);
I = I(:,1:k);
Ynearest = Y(I);
frequencies = histc(Ynearest,1:max(Y),2);
[~,J] = max(frequencies,[],2);
Ypred = C(J);
end
