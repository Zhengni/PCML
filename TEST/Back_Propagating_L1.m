function [delta_W1L,delta_W1R,delta_b1L,delta_b1R ] = Back_Propagating_L1( delta_b2L,delta_b2R, delta_b2LR,W2L,W2R,W2LR,a1L,a1R,xL,xR,H1,d,batch )
%[ delta_W,delta_b ] = Back_Propagating_L3( t,a3,z2 )
%   calculate derivative of E on w1 and b1
%[H2,H1]=size(W2L);
Wl=W2LR(:,1:H1);
Wr=W2LR(:,H1+1:end);

%% left
% size(W2L)
% size(delta_b2L)
% size(Wl)
% size(delta_b2LR)
% size(dif_sigma(2*a1L))
%delta_b1L=4*(W2L'*delta_b2L+Wl'*delta_b2LR).*dif_sigma(2*a1L);
delta_b1L=(W2L'*delta_b2L+Wl'*delta_b2LR).*sech(a1L).^2;
x=zeros(H1,d,batch);
y=zeros(H1,d,batch);
for i=1:batch
    x(:,:,i)=repmat(delta_b1L(:,i),1,d);
    y(:,:,i)=repmat(xL(:,i)',H1,1);
end
delta_W1L=x.*y;
%delta_W1L=delta_b1L*xL';

%% right
delta_b1R=(W2R'*delta_b2R+Wr'*delta_b2LR).*sech(a1R).^2;
x=zeros(H1,d,batch);
y=zeros(H1,d,batch);
for i=1:batch
    x(:,:,i)=repmat(delta_b1R(:,i),1,d);
    y(:,:,i)=repmat(xR(:,i)',H1,1);
end
delta_W1R=x.*y;
%delta_W1R=delta_b1R*xR';

end
