function e= Square_Error(W_1,W_2,W_3,B_1,B_2,B_3,t,x_L,x_R) 
d = size(x_L,1);
N = size(x_L,2);
h1 = size(W_1,2);
h2 = size(W_2,2);
w_L_1 = W_1(1:d,:);
w_R_1 = W_1(d+1:end,:);
w_L_2 = W_2(1:h1,:);
w_R_2 = W_2(h1+1:h1+h1,:);
w_LR_2 = W_2(h1+h1+1:end,:);
w_3 = W_3;
b_L_1 = repmat(B_1(1:h1,:),1,N);
b_R_1 = repmat(B_1(h1+1:end,:),1,N);
b_L_2 = repmat(B_2(1:h2,:),1,N);
b_R_2 = repmat(B_2(h2+1:h2+h2,:),1,N);
b_LR_2 = repmat(B_2(h2+h2+1:end,:),1,N);
b_3 = repmat(B_3,1,N);


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

diff = a_3 - t;
size2 = size(a_3,2);
norm_diff= zeros(1,size2);


for i=1:size2
        norm_diff(i)=norm(diff(i))^2;
end

e = mean(norm_diff);



end