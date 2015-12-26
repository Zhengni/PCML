function result=regulisquar(y,t,w)
dif=y-t;
[sizex,sizey]=size(dif);
normal=zeros(1,sizey);
for i=1:sizey
    normal(i)=norm(dif(:,i));
end
result=mean(normal)/2+norm(w)/2;
    