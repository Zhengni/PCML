% load('norb_5class.mat');
load('init.mat');
load('test_data5.mat');
N = length(train_cat);
N_fold = 10;


plusone = ones(1,N);
%add b to end of x
train_data = [train_data;plusone];

testone = ones(1,length(test_cat_s));
test = [test_left_s_m;test_right_s_m;testone];

% ind0 = find(train_cat==0);
train_cat_r = zeros(5,N);
train_cat_r(1,train_cat==0)=1;
train_cat_r(2,train_cat==1)=1;
train_cat_r(3,train_cat==2)=1;
train_cat_r(4,train_cat==3)=1;
train_cat_r(5,train_cat==4)=1;
ind = 0:N/N_fold:N;


v = 0:0.2:16;
E_reg = zeros(length(v),10);
Et_reg = zeros(length(v),10);
loss_01 = zeros(length(v),10);

%linear classifier squared error
for j=1:length(v)
%     v_i = j+1;
    j
    for i=1:N_fold
        v_data = train_data(:,ind(i)+1:ind(i+1));
        v_cat = train_cat_r(:,ind(i)+1:ind(i+1));
        if i==1
           t_data = train_data(:,ind(i+1)+1:end);
           t_cat = train_cat_r(:,ind(i+1)+1:end);
        elseif i==N_fold
           t_data = train_data(:,1:ind(i));
           t_cat = train_cat_r(:,1:ind(i));
        else
           t_data =  [train_data(:,1:ind(i)),train_data(:,ind(i+1)+1:end)];
           t_cat = [train_cat_r(:,1:ind(i)),train_cat_r(:,ind(i+1)+1:end)];
        end
        
        [t_row t_col]= size(t_data);
        fai = t_data';
        t = t_cat';
        I= eye(t_row);
        w = (fai'*fai+2*v(j)*I)\(fai'*t);
        result = w'*v_data;
        
        [ymax,class_r] = max(result);
        re_m = (result == repmat(ymax,5,1));       %get the position of the largest in the col of validation result 
          E2 = 0.5*trace((result-v_cat)*(result-v_cat)');
%          E2 = 0.5*trace((re_m-v_cat)*(re_m-v_cat)');
         E_reg(j,i) = E2;%+v(j)*trace(w'*w);  
        [~,class_v] = max(v_cat);
        loss_01(j,i) = sum(class_r ~= class_v);
        
        
        
        result_t = w'*t_data; %??
        [max_t, class_t] = max(result_t);
        re_t = (result_t == repmat(max_t,5,1));
        Et = 0.5*trace((result_t-t_cat)*(result_t-t_cat)');
        Et_reg(j,i)=Et;
        
       
        
        
    end
end
    E_reg_a = mean(E_reg,2);
    loss_01_a = mean(loss_01,2);
    
    [val vmin]= min(E_reg_a);  %no.12.  v= 2.2
    dev = std(E_reg(vmin,:));
    error_v = E_reg_a + dev;
    
    
    error_t = mean(Et_reg,2)/9;
    
    %test the test set
    
    

figure;
plot(v,E_reg_a,'r'); %validation error
xlabel('v');% xÖáÃû³Æ
xlim([0,16]);
set(gca,'XTick',0:2:16);
ylabel('Error'); 
hold on;
plot(v,error_v,'g');  %mean+std
hold on;
plot(v,error_t,'b');  %ÊÇ·ñÒª³ı10

% figure;
% plot(v,loss_01_a,'r')




