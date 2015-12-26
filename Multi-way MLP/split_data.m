%%load and change it to double
load('norb_5class.mat');
train_cat_s = double(train_cat_s); 
train_left_s = double(train_left_s);
train_right_s = double(train_right_s);
test_cat_s = double(test_cat_s);
test_left_s = double(test_left_s);
test_right_s = double(test_right_s);

N = length(train_cat_s);
A = randperm(N);
ac_A = A(1:N*2/3);
va_A = A(N*2/3+1:end);


%%
%%preprocessing data
cat_m = train_cat_s;
left_m = mean(mean(train_left_s));
right_m = mean(mean(train_right_s));


%left and right do mean together
l_r_m = 0.5*(left_m+right_m);

%change into zero mean

train_left_s_m = train_left_s-l_r_m;
train_right_s_m = train_right_s-l_r_m;


test_left_s_m = test_left_s-l_r_m;
test_right_s_m = test_right_s-l_r_m;

%variance

train_left_s_var = sqrt(sum(train_left_s_m.^2,2)/N);
train_right_s_var = sqrt(sum(train_right_s_m.^2,2)/N);

%zero mean unite variance data

train_left_s_m = repmat((1./train_left_s_var),1,5400).*train_left_s_m;
train_right_s_m = repmat((1./train_right_s_var),1,5400).*train_right_s_m;


test_left_s_m = repmat((1./train_left_s_var),1,5400).*test_left_s_m;
test_right_s_m = repmat((1./train_right_s_var),1,5400).*test_right_s_m;

%%
%seperate data input 
ac_train_data_cat = cat_m(:,ac_A);
ac_train_data_left = train_left_s_m(:,ac_A);
ac_train_data_right = train_right_s_m(:,ac_A);
va_train_data_cat = cat_m(:,va_A);
va_train_data_left = train_left_s_m(:,va_A);
va_train_data_right = train_right_s_m(:,va_A);

save('splited_data5.mat','ac_train_data_cat','ac_train_data_left','ac_train_data_right','va_train_data_cat','va_train_data_left','va_train_data_right');
save('test_data5.mat','test_cat_s','test_left_s_m','test_right_s_m');

