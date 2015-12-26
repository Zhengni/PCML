function los=Loss(y,t)
[my,py]=max(y);
[mt,pt]=max(t);
error=(py~=pt);
los=mean(error);