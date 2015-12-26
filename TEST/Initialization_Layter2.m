function [ W2, b2 ] = Initialization_Layter2( H1, H2)
%[ W1, b1 ] = Initialization_Layter1( H1, d)
% initialize W1, b1 randomly
W2=normrnd(0,1/(H1+1),H2,H1);
b2=normrnd(0,1/(H1+1),H2,1);

end
