load('test_data5.mat');
load('norb_5class.mat');
N = length(test_cat_s);
% optimal value of v
v =2.2;
plusone = ones(1,N);
%add b to end of x
test_data = [test_left_s_m;test_right_s_m];
t_data = [test_data;plusone];



test_cat_r = zeros(5,N);
test_cat_r(1,test_cat_s==0)=1;
test_cat_r(2,test_cat_s==1)=1;
test_cat_r(3,test_cat_s==2)=1;
test_cat_r(4,test_cat_s==3)=1;
test_cat_r(5,test_cat_s==4)=1;
t_cat = test_cat_r;

[t_row t_col]= size(t_data);
        fai = t_data';
        t = t_cat';
        I= eye(t_row);
        w = (fai'*fai+2*v*I)\(fai'*t);
        result = w'*t_data;
        
        [ymax,class_r] = max(result);
        result = class_r -1;
        result_r = result-test_cat_s;
        ind_wrong = find(result_r~=0);
        ind_right=find(result_r ==0);
% w_data = test_data(:,ind_wrong(1));
% r_data = test_data(:,ind_ringht(1));
w_data_l = reshape(test_left_s(:,ind_wrong(1)),24,[]);
w_data_r = reshape(test_right_s(:,ind_wrong(1)),24,[]);
r_data_l = reshape(test_left_s(:,ind_right(1)),24,[]);
r_data_r = reshape(test_right_s(:,ind_right(1)),24,[]);
figure(1);
subplot(2,2,1);imshow(w_data_l);title('wrong data left camera')
subplot(2,2,2);imshow(w_data_r);title('wrong data right camera')
subplot(2,2,3);imshow(r_data_l);title('right data left camera')
subplot(2,2,4);imshow(r_data_r);title('wrong data right camera')
        
        