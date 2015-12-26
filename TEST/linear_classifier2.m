clc;clear all;

load norb_5class_done.mat;
[sizei,sizen]=size(train1_left_s);
sizeim=2*sizei+1;
train_image=ones(sizeim,sizen);
train_image(1:sizei,:)=train1_left_s;
train_image(sizei+1:2*sizei,:)=train1_right_s;
t_mu=zeros(5,sizen);
t=train1_cat_s;
A=1:sizen;
for i=1:sizen
    t_mu(t(i)+1,i)=1;
end

batch=1;
ita=0.002;
mu=0.5;

[sizeti,sizetn]=size(test_left_s);
sizetim=2*sizeti+1;
test_image=ones(sizetim,sizetn);
test_image(1:sizeti,:)=test_left_s;
test_image(sizeti+1:2*sizeti,:)=test_right_s;
tt_mu=zeros(5,sizetn);
tt=test_cat_s;

for i=1:sizetn
    tt_mu(tt(i)+1,i)=1;
end

[sizevi,sizevn]=size(valid_left_s);
sizevim=2*sizevi+1;
valid_image=ones(sizevim,sizevn);
valid_image(1:sizevi,:)=valid_left_s;
valid_image(sizevi+1:2*sizevi,:)=valid_right_s;
v_mu=zeros(5,sizevn);
v=valid_cat_s;
for i=1:sizevn
    v_mu(v(i)+1,i)=1;
end

w=normrnd(0,1/sizevim,sizevim,5); %initianlize w
old_w=0;
loop=200;
v=zeros(1,loop);
tr=zeros(1,loop);
for k=1:loop
    k
    y_test=w'*test_image;
    y_train=w'*train_image;
    %los=Loss(y_test,tt_mu);
    los=LogError(y_test,tt_mu);
    v(k)=los; %validation error
%     tr(k)=Loss(y_train,t_mu);
tr(k)=LogError(y_train,t_mu);
    l=randperm(3600);
for i=1:3600
    random=l(i);
    %random=randi([1 3600],batch,1);
    x_image=train_image(:,random);
    x_t=t_mu(:,random);
    y=w'*x_image;
    delta_w=zeros(sizevim,5,batch);
    for j=1:batch
        delta_w(:,:,j)=((delta_lsexp(y(:,j))-x_t(:,j))*x_image(:,j)')';
    end
    delta_w=mean(delta_w,3);
    w=w-ita*(1-mu)*delta_w+mu*old_w;
    old_w=-ita*(1-mu)*delta_w+mu*old_w;
end
end
figure(1);
plot(tr);
figure(2);
plot(v,'r');
    
