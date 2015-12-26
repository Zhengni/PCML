function [ a1L,a1R,z1L,z1R] = Forward_Propagating_L1( W1L,W1R,b1L,b1R,xL,xR )
%Layer 1 forward propagation to get activation functions and transfer
%functions with respect to given weights

%%   get size H1,d
[H1,d]=size(W1L);
[d,batch]=size(xL);

b1L=repmat(b1L,1,batch);
b1R=repmat(b1R,1,batch);

%% get activation functions a for left and right classes
a1L=W1L*xL+b1L;
a1R=W1R*xR+b1R;

%% get transfer function z for left and right classes
z1L=tanh(a1L);
z1R=tanh(a1R);

end

