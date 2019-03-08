% function that adds time 1 for the next coming point and identify direction
% of id
function [T,id]=addtimenext2(D,T,i,j,n,m,id)
if j>1
    if D(i,j-1)==1
        T(i,j-1)=T(i,j)+1;
        id(1)=1;
    else
        id(1)=0;
    end
end
if i>1
	if D(i-1,j)==2
        T(i-1,j)=T(i,j)+1;
        id(2)=1;
    else
        id(2)=0;
	end
end
if j<m
	if D(i,j+1)==3
        T(i,j+1)=T(i,j)+1;
        id(3)=1;
    else
        id(3)=0;
	end
end
if i<n
	if D(i+1,j)==4
        T(i+1,j)=T(i,j)+1;
        id(4)=1;
    else
        id(4)=0;
	end
end

