clear;
load('logistic_error_train_data.mat');
load('logistic_error_test_data.mat');

%add one to tail of data
ac_train_data = [ac_train_data;ones(1,size(ac_train_data,2))];
va_train_data = [va_train_data;ones(1,size(va_train_data,2))];
test_data = [test_data;ones(1,size(test_data,2))];

[N,L] = size(ac_train_data);
%initial w
w = normrnd(0,1/N,N,5);
d_w = zeros(size(w));
%initial learning rate and momentum term coefficient (0~1)
n = 0.02; %learn
u = 0.4;

%set number of data using to update w each loop
N_u = 3600;
N_j = 200;

Elog_train = zeros(1,N_j);
Elog_validation = zeros(1,N_j);
Elog_test = zeros(1,N_j);
E01_va = zeros(1,N_j);
for j =1:N_j
    
    Elog_train(j)  = compute_logistic_error(ac_train_data,ac_train_data_cat,w);
%compute logistic error over validation data 
Elog_validation(j) = compute_logistic_error(va_train_data,va_train_data_cat,w);
%comput logistic error over test data
Elog_test(j) = compute_logistic_error(test_data,test_cat_s,w);

E01_va(j) = compute_01_error(va_train_data,va_train_data_cat,w);

for i = 1:N_u:L
%compute logistic error over train data 

%update w
update_data = ac_train_data(:,i:i+N_u-1);
updata_cat = ac_train_data_cat(:,i:i+N_u-1);
y_update = w'*update_data;
t_r = zeros(5,size(updata_cat,2));
t_r(1,updata_cat==0)=1;
t_r(2,updata_cat==1)=1;
t_r(3,updata_cat==2)=1;
t_r(4,updata_cat==3)=1;
t_r(5,updata_cat==4)=1;
g_w = (((exp(y_update)./repmat(sum(exp(y_update)),5,1)-t_r)*update_data')/N_u)';
d_w = -n*(1-u)*g_w+u*d_w;
w = w+d_w;
end
end

test = w'* test_data;
deviation =std(Elog_test);


figure;

plot(Elog_train,'b');
figure;
plot(Elog_validation,'r');
title('learning rate 0.001');
%plot(Elog_test,'g');
figure
plot(E01_va,'g');


