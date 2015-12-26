function [ delta_W3,delta_b3 ] = Back_Propagating_L3( t,a3,z2,H3,H2,batch )
%[ delta_W,delta_b ] = Back_Propagating_L3( t,a3,z2 )
% calculate derivative of E on w3 and b3 -(t*exp(-t*x))/(exp(-t*x) + 1)
delta_b3=a3-t;% H3*batch

delta_W3=zeros(H3,H2,batch);% H2*batch

for i=1:batch
    delta_W3(:,:,i)=delta_b3(:,i)*z2(:,i)';
end

end

