function  [Terror]=trainError(W1L,W1R,b1L,b1R,W2L,W2R,W2LR,b2L,b2R,b2LR,W3,b3,tL,tR,t)

%calculate error in validation set
    [ a1L,a1R,z1L,z1R] = Forward_Propagating_L1( W1L,W1R,b1L,b1R,tL,tR );
    [ a2L,a2R,a2LR,z2] = Forward_Propagating_L2( W2L,W2R,W2LR,b2L,b2R,b2LR,z1L,z1R );
    [ a3] = Forward_Propagating_L3( W3,b3,z2 );
    [sizex,sizey]=size(a3);
    diff=a3-t;
    norm_diff=zeros(1,sizey);
    for i=1:sizey
        norm_diff(i)=norm(diff(i))^2;
    end
    Terror=mean(norm_diff);
end