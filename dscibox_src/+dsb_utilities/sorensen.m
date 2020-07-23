function distance = sorensen(a,b)
distance = sum(abs(a-b))/sum((a+b))
end
