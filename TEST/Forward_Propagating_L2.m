function [ a2L,a2R,a2LR,z2] = Forward_Propagating_L2( W2L,W2R,W2LR,b2L,b2R,b2LR,z1L,z1R )
%Layer 1 forward propagation to get activation functions and transfer
%functions with respect to given weights

%%   get size H1,H2, get z1LR
[H2,H1]=size(W2L);
z1LR=[z1L; z1R];
[d,batch]=size(z1L);

b2L=repmat(b2L,1,batch);
b2R=repmat(b2R,1,batch);
b2LR=repmat(b2LR,1,batch);
%% get activation functions a for left and right classes
%   size(z1L)
%     size(W2L)
%  size(b2L)
a2L=W2L*z1L+b2L;
a2R=W2R*z1R+b2R;
a2LR=W2LR*z1LR+b2LR;

%% get transfer function z for left and right classes
z2=a2LR.*sigma(a2L).*sigma(a2R);


end

