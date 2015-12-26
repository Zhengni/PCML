function  E_01 = compute_01_error(x,t,w)

%change category
t_r = zeros(5,size(t,2));
t_r(1,t==0)=1;
t_r(2,t==1)=1;
t_r(3,t==2)=1;
t_r(4,t==3)=1;
t_r(5,t==4)=1;

% comput logistic error
y = w'*x;
error = zeros(1,1800);

for i=1:size(y,2)
    %map y to 0 and 1
    [val ind] = max(y(:,i));
    y(:,i)=0;
    y(ind,i)=1;
    error(i) = length(find(t_r(:,i)~=y(:,i))); %if error =2,else =0
end

E_01 = length(find(error ~= 0));


end