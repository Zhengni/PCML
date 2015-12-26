function [ delta_W2L,delta_b2L,delta_W2R,delta_b2R,delta_W2LR,delta_b2LR ] = Back_Propagating_L2( W3,delta_b3, z1L,z1R,a2L,a2R,a2LR,H1,H2,batch )
%[ delta_W,delta_b ] = Back_Propagating_L3( t,a3,z2 )
%   calculate derivative of E on w2 and b2
z1LR=[z1L;z1R];
%% left
% size(z1LR)
% size(W3)
% size(a2L)
W3=W3'; % H3*H2 --> H2*H3
% delta_b3 --> H3*batch

delta_b2L=(W3*delta_b3).*dif_sigma(a2L).*sigma(a2R).*a2LR;
x=zeros(H2,H1,batch);
y=zeros(H2,H1,batch);
for i=1:batch
    x(:,:,i)=repmat(delta_b2L(:,i),1,H1);
    y(:,:,i)=repmat(z1L(:,i)',H2,1);
end
delta_W2L=x.*y;

%% right
delta_b2R=(W3*delta_b3).*dif_sigma(a2R).*sigma(a2L).*a2LR;
x=zeros(H2,H1,batch);
y=zeros(H2,H1,batch);
for i=1:batch
    x(:,:,i)=repmat(delta_b2R(:,i),1,H1);
    y(:,:,i)=repmat(z1R(:,i)',H2,1);
end
delta_W2R=x.*y;

%% LR
delta_b2LR=(W3*delta_b3).*sigma(a2L).*sigma(a2R);
x=zeros(H2,2*H1,batch);
y=zeros(H2,2*H1,batch);
for i=1:batch
    x(:,:,i)=repmat(delta_b2LR(:,i),1,2*H1);
    y(:,:,i)=repmat(z1LR(:,i)',H2,1);
end
delta_W2LR=x.*y;

end

