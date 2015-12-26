function [ W1, b1 ] = Initialization_Layter1( H1, d)
%[ W1, b1 ] = Initialization_Layter1( H1, d)
% initialize W1, b1 randomly
W1=normrnd(0,1/(d+1),H1,d);

b1=normrnd(0,1/(d+1),H1,1);


end

