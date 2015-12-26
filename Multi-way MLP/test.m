clear all
clc

load('splited_data5.mat')
load('test_data5.mat')
d_L = size(ac_train_data_left,1);
d_R = size(ac_train_data_right,1);
Lt = length(ac_train_data_cat);

Lv = length(va_train_data_cat);
Length = length(test_cat_s);
%set the number of hidden units
h1 = 50;    %task 2 use environ 50
h2 = 30;
h3= 5;      %five category

%transform the category 
t_r = zeros(5,Lt);

ind0 = find(ac_train_data_cat==0);
ind1 = find(ac_train_data_cat==1);
ind2 = find(ac_train_data_cat==2);
ind3 = find(ac_train_data_cat==3);
ind4 = find(ac_train_data_cat==4);

t_r(1,ind0)=1;
t_r(2,ind1)=1;
t_r(3,ind2)=1;
t_r(4,ind3)=1;
t_r(5,ind4)=1;

t_v = zeros(5,Lv);

index0 = find(va_train_data_cat==0);
index1 = find(va_train_data_cat==1);
index2 = find(va_train_data_cat==2);
index3 = find(va_train_data_cat==3);
index4 = find(va_train_data_cat==4);


t_v(1,index0)=1;
t_v(2,index1)=1;
t_v(3,index2)=1;
t_v(4,index3)=1;
t_v(5,index4)=1;


t_t = zeros(5,Length);

inde0 = find(test_cat_s==0);
inde1 = find(test_cat_s==1);
inde2 = find(test_cat_s==2);
inde3 = find(test_cat_s==3);
inde4 = find(test_cat_s==4);

t_t(1,inde0)=1;
t_t(2,inde1)=1;
t_t(3,inde2)=1;
t_t(4,inde3)=1;
t_t(5,inde4)=1;

%initial w and b

[W_1,W_2,W_3,B_1,B_2,B_3] = Initial_MLP(d_L,d_R,h1,h2,h3);
d_W_1 = zeros(size(W_1));
d_W_2 = zeros(size(W_2));

d_W_3 = zeros(size(W_3));
d_B_1 = zeros(size(B_1));
d_B_2 = zeros(size(B_2));
d_B_3 = zeros(size(B_3));



 %everytime update one from a_3_q


%set the number of data used to compute gradient decent
k = 1;

loop = 30;
L_T = zeros(1,loop);
L_V = zeros(1,loop);
P_V = zeros(1,loop);

Test = zeros(1,loop);
estimation = zeros(1,loop);

for j =1:loop
    j
    L_T(j) = Square_Error(W_1,W_2,W_3,B_1,B_2,B_3,t_r,ac_train_data_left,ac_train_data_right); 

    L_V(j) = Square_Error(W_1,W_2,W_3,B_1,B_2,B_3,t_v,va_train_data_left,va_train_data_right);

    [Test(j),estimation] = Test_Square_Error(W_1,W_2,W_3,B_1,B_2,B_3,t_t,test_left_s_m,test_right_s_m);
    
    P_V(j) = Pattern_Error(W_1,W_2,W_3,B_1,B_2,B_3,t_v,va_train_data_left,va_train_data_right);
    for i = 1:k:Lt
 
 
    %choose data to compute Error and gradient decent
    x_L = ac_train_data_left(:,i:i+k-1);
    x_R = ac_train_data_right(:,i:i+k-1);
    t = t_r(:,i:i+k-1);

    ind = find(t==1);

    %compute gradient decent
    [GR_w_3,GR_w_2,GR_w_1,GR_b_3,GR_b_2,GR_b_1]=Gradient_decent(W_1,W_2,W_3,B_1,B_2,B_3,t,x_L,x_R);

    %set the momentum term coefficient (between 0 to 1)
    u = 0.4;
    %set the learning rate
    n = 0.0003;




    %update the w and b
    d_W_1 = -n*(1-u)*GR_w_1+u*d_W_1;
    d_W_2 = -n*(1-u)*GR_w_2+u*d_W_2;
    d_W_3 = -n*(1-u)*GR_w_3+u*d_W_3;
    d_B_1 = -n*(1-u)*GR_b_1+u*d_B_1;
    d_B_2 = -n*(1-u)*GR_b_2+u*d_B_2;
    d_B_3 = -n*(1-u)*GR_b_3+u*d_B_3;
    % 


    W_1 = W_1+d_W_1;
    W_2 = W_2+d_W_2;
    W_3= W_3+d_W_3;
    B_1 = B_1+d_B_1;
    B_2 = B_2+d_B_2;
    B_3= B_3+d_B_3;



    end


end

a3 = a3_test(W_1,W_2,W_3,B_1,B_2,B_3,t_t,test_left_s_m,test_right_s_m);
[maxt indt] = max(t_t);
[maxa inda] = max(a3);
CM = confusionmat(indt,inda);

figure;

plot(L_T,'b');
hold on;
plot(L_V,'r');

figure;
plot(P_V,'g');
% figure(2);
% plot(Test,'g');

