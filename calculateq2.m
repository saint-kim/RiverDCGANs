% function calculates q
% getxywid
% addtimenext2
function [FA,T,q]=calculateq2(D,I,n,m,n0,m0)

% Compute Flow Accumulation Matrix FA
FA=zeros(n,m);
for i=1:n
    for j=1:m
        if j>1
            if D(i,j-1)==1
                FA(i,j)=I(i,j-1);
            end
        end
        if i>1
            if D(i-1,j)==2
                FA(i,j)=FA(i,j)+I(i-1,j);
            end
        end
        if j<m
            if D(i,j+1)==3
                FA(i,j)=FA(i,j)+I(i,j+1);
            end
        end
        if i<n
            if D(i+1,j)==4
                FA(i,j)=FA(i,j)+I(i+1,j);
            end
        end
    end
end

% Compute travel time matrix T
T=zeros(n,m);
T(n0,m0)=1; % Set travel time at the outlet 1
%nostpts=1; %number of start point
x(1)=n0;y(1)=m0; % Initial location of a start point at the oulet
id(1:4)=0; % indicator showing that flow exists
minT=0;

% add travel time to incoming node from starting point
% this fn add travel time to incoming node 
[T,id]=addtimenext2(D,T,x(1),y(1),n,m,id); 
nonewpts=sum(id);
nostpts=nonewpts;
    
% get x and y coordinates for new starting points
    [xnew,ynew]=getxywid(x,y,x(1),y(1),id);

tmax=n*m;
tacc=0;
while minT==0
    % now we have nostpts starting points
    x=xnew;y=ynew;
    idi=zeros(nostpts,4);
    for k=1:nostpts
        %  k=1;
        id=idi(k,:);
        [T,id]=addtimenext2(D,T,x(k),y(k),n,m,id);
        idi(k,:)=id;
        if k==1
            [xnew,ynew]=getxywid(x,y,x(k),y(k),id);
% take exactly as long as the length of id otherwise we have extra info
% from previous time step
            xnew=xnew(1:sum(id)); 
            ynew=ynew(1:sum(id)); 
        else
            [xtemp,ytemp]=getxywid(x,y,x(k),y(k),id);
            xtemp=xtemp(1:sum(id));
            ytemp=ytemp(1:sum(id));
            xnew=[xnew xtemp];
            ynew=[ynew ytemp];
        end
    end
    nostpts=length(xnew);

    % check minT is equal to 0 if not, exit
    tacc=tacc+1; %check if looping
    minT=min(min(T));
    if tacc>tmax
        'there exists a loop';
        break
    end

end



% compute flow along time
maxT=max(max(T));
q=zeros(1,maxT);
for i=1:n
    for j=1:m
        for k=1:maxT
            if T(i,j)==k
                q(k)=q(k)+FA(i,j);
            end
        end
    end
end
q=[0 1*I(n,m) q(1:maxT-1) 0];
