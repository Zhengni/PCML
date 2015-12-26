function [W_1,W_2,W_3,B_1,B_2,B_3]= Initial_MLP(dl,dr,h1,h2,h3)
%initializ
d = dl+dr;

w_L_1 = randn(dl,h1)/d;%every column vector stands for a wl,every wl has d1 components  //dimension should be d1?
w_R_1 = randn(dr,h1)/d;

b_L_1 = randn(h1,1)/d;%every row stands for a bl
b_R_1 = randn(h1,1)/d;

w_L_2 = randn(h1,h2)/h1;%every column vector stands for a wl,every wl has d1 components
w_R_2 = randn(h1,h2)/h1;
w_LR_2 = randn(2*h1,h2)/2/h1;%every column vector stands for a wl,every wlr has d components  //dimesnion 2h1?

b_L_2 = randn(h2,1)/h1;%every row stands for a bl
b_R_2 = randn(h2,1)/h1;
b_LR_2 = randn(h2,1)/h1;

w_3 = randn(h2,h3)/h2;
b_3 = randn(h3,1)/h2;

%output
W_1 = [w_L_1;w_R_1];
W_2 = [w_L_2;w_R_2;w_LR_2];
W_3 = w_3;
B_1 = [b_L_1;b_R_1];
B_2 = [b_L_2;b_R_2;b_LR_2];
B_3= b_3;
end