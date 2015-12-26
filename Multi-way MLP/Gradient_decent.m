function [GR_w_3,GR_w_2,GR_w_1,GR_b_3,GR_b_2,GR_b_1]= Gradient_decent(W_1,W_2,W_3,B_1,B_2,B_3,t,x_L,x_R)
% extract Ws and Bs from input
d = size(x_L,1);
L = size(x_L,2);
h1 = size(W_1,2);
h2 = size(W_2,2);
h3 = size(W_3,2);
w_L_1 = W_1(1:d,:);
w_R_1 = W_1(d+1:end,:);
w_L_2 = W_2(1:h1,:);
w_R_2 = W_2(h1+1:h1+h1,:);
w_LR_2 = W_2(h1+h1+1:end,:);
w_3 = W_3;
%adjust b to input data
b_L_1 = repmat(B_1(1:h1,:),1,L);
b_R_1 = repmat(B_1(h1+1:end,:),1,L);
b_L_2 = repmat(B_2(1:h2,:),1,L);
b_R_2 = repmat(B_2(h2+1:h2+h2,:),1,L);
b_LR_2 = repmat(B_2(h2+h2+1:end,:),1,L);
b_3 = repmat(B_3,1,L);



% forward pass
a_L_1 = w_L_1.'*x_L+b_L_1;
a_R_1 = w_R_1.'*x_R+b_R_1;
z_L_1 = tanh(a_L_1);
z_R_1 = tanh(a_R_1);
z_LR_1 = [z_L_1;z_R_1];
a_L_2 = w_L_2.'*z_L_1+b_L_2;
a_R_2 = w_R_2.'*z_R_1+b_R_2;
a_LR_2 = w_LR_2.'*z_LR_1+b_LR_2;
z_2 = a_LR_2./((1+exp(-a_L_2)).*(1+exp(-a_R_2)));
a_3 = w_3.'*z_2+b_3;


% compute the output residual
% n = d;
r_3 = a_3 - t; %square error
dz1 = a_LR_2.*exp(-a_L_2)./((1+exp(-a_R_2)).*((1+exp(-a_L_2)).^2));    
dz2 = a_LR_2.*exp(-a_R_2)./((1+exp(-a_L_2)).*((1+exp(-a_R_2)).^2));
dz3 = 1./((1+exp(-a_R_2)).*(1+exp(-a_L_2))); 


r_L_2 = w_3*r_3.*dz1;    %r_2 is a vector
r_R_2 = w_3*r_3.*dz2; 
r_LR_2 = w_3*r_3.*dz3;
r_L_1 = (w_L_2*r_L_2+w_LR_2(1:h1,:)*r_LR_2).*(1-z_L_1.^2); %公式有点推的不一样
r_R_1 = (w_L_2*r_R_2+w_LR_2(h1+1:end,:)*r_LR_2).*(1-z_R_1.^2);



% back propagation
gr_w_3 = (r_3*z_2').'/L;
gr_w_L_2 = (r_L_2*z_L_1.').'/L;%every column vector is a gradient for a w_2
gr_w_R_2 = (r_R_2*z_R_1.').'/L;
gr_w_LR_2 = (r_LR_2*[z_L_1;z_R_1].').'/L;
gr_w_L_1 = (r_L_1*x_L.').'/L;
gr_w_R_1 = (r_R_1*x_R.').'/L;
gr_b_3 = mean(r_3,2);
gr_b_L_2 = mean(r_L_2,2);
gr_b_R_2 = mean(r_R_2,2);
gr_b_LR_2 = mean(r_LR_2,2);
gr_b_L_1 = mean(r_L_1,2);
gr_b_R_1 = mean(r_R_1,2);


%output
GR_w_3 = gr_w_3;
GR_w_2 = [gr_w_L_2;gr_w_R_2;gr_w_LR_2];
GR_w_1 = [gr_w_L_1;gr_w_R_1];
GR_b_3 = gr_b_3;
GR_b_2 = [gr_b_L_2;gr_b_R_2;gr_b_LR_2];
GR_b_1 = [gr_b_L_1;gr_b_R_1];

