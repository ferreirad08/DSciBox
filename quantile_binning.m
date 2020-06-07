function xt = quantile_binning(x,n_bins)
%Author: David Ferreira - Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%Quantile Binning Transformation
%
%Syntax
%1. xt = quantile_binning(x,n_bins)
%
%Description
%1. Discretize the data of a vector based on quantiles.
%
%x is a variable with continuous values.
%n_bins is the number of groupings.
%
%Examples
%1.
%     x = [30, 64, 49, 26, 69, 23, 56, 7, 69, 67,...
%         87, 14, 67, 33, 88, 77, 75, 47, 44, 93];
%     n_bins = 10;
%     xt = quantile_binning(x,n_bins)
%     xt =
%         2 5 4 1 7 1 4 0 7 6 8 0 6 2 9 8 7 3 3 9

if ~isrow(x), x = x'; end

p = (1:n_bins-1)/n_bins;
Q = quantile_analysis(x,p);
xt = sum(repmat(x,numel(Q),1)>=repmat(Q,1,(numel(x))));
end
