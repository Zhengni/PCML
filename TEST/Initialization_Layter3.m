function [ W3, b3 ] = Initialization_Layter3( H2,H3)
%[ W1, b1 ] = Initialization_Layter1( H1, d)
% initialize W1, b1 randomly
W3=normrnd(0,1/(H2+1),H3,H2);
b3=normrnd(0,1/(H2+1),H3,1);

end
