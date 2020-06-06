function xt = binning(x,n_bins)
    if ~isrow(x), x = x'; end
    
    p = (1:n_bins-1)/n_bins;
    Q = quantile(x,p);
    xt = sum(x>=Q);
end
