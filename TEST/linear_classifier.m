clc;clear all;

load norb_5class_done1.mat;
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

[sizeti,sizetn]=size(test_left_s);
sizetim=2*sizeti+1;
test_image=ones(sizetim,sizetn);
test_image(1:sizeti,:)=test_left_s;
test_image(sizeti+1:2*sizeti,:)=test_right_s;
tt_mu=zeros(5,sizetn);
tt=test_cat_s;
A=1:sizen;
for i=1:sizetn
    tt_mu(tt(i)+1,i)=1;
end

bound=40;
total_los=zeros(1,bound);
%random=randperm(5400);
%train_image=train_image(:,random);
%t_mu=t_mu(:,random);

for v_1=1:bound
    v=(v_1-5)/5;
    for i=1:10
        start=540*(i-1)+1;
        finish=540*i;
        B=start:finish;
        used=setdiff(A,B);
        wv=(train_image(:,used)*train_image(:,used)'+v*eye(sizeim))\train_image(:,used)*t_mu(:,used)';
        y=wv'*train_image(:,B);
        los=Loss(y,t_mu(:,B));
        total_los(v_1)=total_los(v_1)+los;
    end
    total_los(v_1)=total_los(v_1)/10;
end
%error=zeros(1,500);
%for i=1:500
%v=30;
%v=i;
%wv=(train_image*train_image'+v*eye(sizeim))\train_image*t_mu';
%y=wv'*test_image;
%los=Loss(y,tt_mu);
%error(i)=los;
%end
x1=1:bound;
x=(x1-5)/5;
plot(x,total_los);
