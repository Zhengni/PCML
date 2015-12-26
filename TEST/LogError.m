function result=LogError(y,t)
    [sizex,sizey]=size(y);
    x=exp(y);
    lsexp=sum(x,1);
    lsexp=log(lsexp);
    t1=t';
    ty=zeros(1,sizey);
    for i=1:sizey
        ty(i)=t1(i,:)*y(:,i);
    end
    error=lsexp-ty;
    result=mean(error);
    