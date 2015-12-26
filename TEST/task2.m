clc;clear all;

load splited_data5.mat;

[sizei,sizen]=size(ac_train_data_left);
[sizevi,sizevn]=size(va_train_data_left);

vt=va_train_data_cat;
vxL=va_train_data_left;
vxR=va_train_data_right;

tt=ac_train_data_cat;
txL=ac_train_data_left;
txR=ac_train_data_right;


vt_mu=zeros(5,sizevn);
tt_mu=zeros(5,sizen);


for i=1:sizen
    tt_mu(tt(i)+1,i)=1;   %map category to 5d
end

for i=1:sizevn
    vt_mu(vt(i)+1,i)=1;
end

d=576;
H1=50;
H2=30;
H3=5; %cannot change this value, for output of a3 is 5-dimension
batch=1;
ita=0.01;
mu=0.5;

old_W1L=0;
old_W1R=0;
old_b1L=0;
old_b1R=0;

old_W2L=0;
old_W2R=0;
old_W2LR=0;
old_b2L=0;
old_b2R=0;
old_b2LR=0;

old_W3=0;
old_b3=0;

%% Initialization according to number of units of each layer

[ W1L, b1L ] = Initialization_Layter1( H1, d);
[ W1R, b1R ] = Initialization_Layter1( H1, d);
[ W2L, b2L ] = Initialization_Layter2( H1, H2);
[ W2LR, b2LR ] = Initialization_Layter2( 2*H1, H2);
[ W2R, b2R ] = Initialization_Layter2( H1, H2);
[ W3, b3 ] = Initialization_Layter3( H2,H3);

%% MLP
loop=100;
v=zeros(1,loop);
tr=zeros(1,loop);
te=zeros(1,loop);
for i=1:loop
    i
    Terror=trainError(W1L,W1R,b1L,b1R,W2L,W2R,W2LR,b2L,b2R,b2LR,W3,b3,txL,txR,tt_mu);

    tr(i)=Terror;

    Verror=validatError(W1L,W1R,b1L,b1R,W2L,W2R,W2LR,b2L,b2R,b2LR,W3,b3,vxL,vxR,vt_mu);
    
    v(i)=Verror;
    
    
    
    y=randperm(3600);
    for j=1:3600
    %random=randi([1,sizen],batch,1);
    random=y(j);
    xL=txL(:,random);
    xR=txR(:,random);
    t=tt_mu(:,random);
    
    %calculate output a3
    [ a1L,a1R,z1L,z1R] = Forward_Propagating_L1( W1L,W1R,b1L,b1R,xL,xR );
    [ a2L,a2R,a2LR,z2] = Forward_Propagating_L2( W2L,W2R,W2LR,b2L,b2R,b2LR,z1L,z1R );
    [ a3] = Forward_Propagating_L3( W3,b3,z2 );
    
    [ delta_W3,delta_b3 ] = Back_Propagating_L3( t,a3,z2,H3,H2,batch );
    [ delta_W2L,delta_b2L,delta_W2R,delta_b2R,delta_W2LR,delta_b2LR ] = Back_Propagating_L2( W3,delta_b3, z1L,z1R,a2L,a2R,a2LR,H1,H2,batch );
    [ delta_W1L,delta_W1R,delta_b1L,delta_b1R ] = Back_Propagating_L1( delta_b2L,delta_b2R, delta_b2LR,W2L,W2R,W2LR,a1L,a1R,xL,xR,H1,d,batch );
    
    %% update--stochastic gradient descent with momentem term
    % layer1
    delta_b1L=mean(delta_b1L,2);
    delta_b1R=mean(delta_b1R,2);
    delta_W1L=mean(delta_W1L,3);
    delta_W1R=mean(delta_W1R,3);
    W1L=W1L-ita*(1-mu)*delta_W1L+mu*old_W1L;
    W1R=W1R-ita*(1-mu)*delta_W1R+mu*old_W1R;
    b1L=b1L-ita*(1-mu)*delta_b1L+mu*old_b1L;
    b1R=b1R-ita*(1-mu)*delta_b1R+mu*old_b1R;
    % layer2
    delta_b2L=mean(delta_b2L,2);
    delta_b2R=mean(delta_b2R,2);
    delta_b2LR=mean(delta_b2LR,2);
    delta_W2L=mean(delta_W2L,3);
    delta_W2R=mean(delta_W2R,3);
    delta_W2LR=mean(delta_W2LR,3);
    W2L=W2L-ita*(1-mu)*delta_W2L+mu*old_W2L;
    W2R=W2R-ita*(1-mu)*delta_W2R+mu*old_W2R;
    b2L=b2L-ita*(1-mu)*delta_b2L+mu*old_b2L;
    b2R=b2R-ita*(1-mu)*delta_b2R+mu*old_b2R;
    W2LR=W2LR-ita*(1-mu)*delta_W2LR+mu*old_W2LR;
    b2LR=b2LR-ita*(1-mu)*delta_b2LR+mu*old_b2LR;
    % layer3
    delta_b3=mean(delta_b3,2);
    delta_W3=mean(delta_W3,3);
    W3=W3-ita*(1-mu)*delta_W3+mu*old_W3;
    b3=b3-ita*(1-mu)*delta_b3+mu*old_b3;

    %% remember old update
    old_W1L=-ita*(1-mu)*delta_W1L+mu*old_W1L;
    old_W1R=-ita*(1-mu)*delta_W1L+mu*old_W1R;
    old_b1L=-ita*(1-mu)*delta_b1L+mu*old_b1L;
    old_b1R=-ita*(1-mu)*delta_b1L+mu*old_b1R;
    
    old_W2L=-ita*(1-mu)*delta_W2L+mu*old_W2L;
    old_W2R=-ita*(1-mu)*delta_W2L+mu*old_W2R;
    old_W2LR=-ita*(1-mu)*delta_W2LR+mu*old_W2LR;
    old_b2L=-ita*(1-mu)*delta_b2L+mu*old_b2L;
    old_b2R=-ita*(1-mu)*delta_b2L+mu*old_b2R;
    old_b2LR=-ita*(1-mu)*delta_b2L+mu*old_b2LR;

    old_W3=-ita*(1-mu)*delta_W3+mu*old_W3;
    old_b3=-ita*(1-mu)*delta_b3+mu*old_b3;
    end
end

figure(1);
plot(tr);
figure(2);
plot(v,'r');
