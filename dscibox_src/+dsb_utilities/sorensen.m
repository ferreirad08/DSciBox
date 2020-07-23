function D = sorensen(A,B)
D = sum(abs(A-B),2) ./ sum(A+B,2);
end
