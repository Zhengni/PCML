%init
load('norb_5class.mat');

data_cat_s = double(train_cat_s); 
data_left_s = double(train_left_s);
data_right_s = double(train_right_s);


train_data = [data_left_s;data_right_s];
N = length(train_cat_s);
A = randperm(N);
data_cat_s = data_cat_s(A);
train_data = train_data(:,A);

%%
%%preprocessing data
% cat_m = mean(train_cat_s);
data_m = mean(mean(train_data));

%change into zero mean
% train_cat_s_m = train_cat_s-cat_m;
train_data_m = train_data-data_m;

%variance
% train_cat_s_var = sqrt(sum(train_cat_s_m.^2)/N);
train_data_var = sqrt(sum(train_data_m.^2,2)/(2*N));


%zero mean unite variance data
% train_cat_s_m = (1/train_cat_s_var)*train_cat_s_m;

train_data = repmat((1./train_data_var),1,N).*train_data_m;
train_cat = data_cat_s;

save('init.mat','train_data','train_cat');