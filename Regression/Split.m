%split into 10 folder
function [x y z u] = Split(group,data_cat_s,data)
N = length(data_cat_s);

u = data_cat_s(:,group*N/10+1:(group+1)*N/10);

data_cat_s(:,group*N/10+1:(group+1)*N/10) = [];%и╬ЁЩ1/10ап

y = data_cat_s;

z= data(:,group*N/10+1:(group+1)*N/10);

data(:,group*N/10+1:(group+1)*N/10) = [];

x=data; %train_data deleted




