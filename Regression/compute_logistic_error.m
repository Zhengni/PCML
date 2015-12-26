function E_log = compute_logistic_error(x,t,w)

%change category
t_r = zeros(5,size(t,2));
t_r(1,t==0)=1;
t_r(2,t==1)=1;
t_r(3,t==2)=1;
t_r(4,t==3)=1;
t_r(5,t==4)=1;

% comput logistic error
y = w'*x;
E_log = mean(log(sum(exp(y),1))-diag(t_r'*y)');



end