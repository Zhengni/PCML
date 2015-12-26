function [ a3] = Forward_Propagating_L3( W3,b3,z2 )
%Layer 1 forward propagation to get activation functions and transfer
%functions with respect to given weights

%%   get size H3,H2
[H2,H3]=size(W3);
[d,batch]=size(z2);
% size(b3)

b3=repmat(b3,1,batch);
% size(W3)
% size(z2)
% size(b3)
%% get activation functions a for left and right classes
a3=W3*z2+b3;

end


