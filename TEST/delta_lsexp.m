function result=delta_lsexp(y)
    x=exp(y);

    sumexp=sum(x);
    
    result=x/sumexp;
end
    
