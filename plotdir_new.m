%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot network D
% text wts on D 
% the width varies proportional to wts (what to show)
% 04/17/10 Yongwon Seo
% pre-define figure size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotdir_new(n,m,D,wts,show,sz1,sz2)
% function plotdir3(n,m,D,wts,show,model_title)

[x,y] = meshgrid(1:1:m,1:1:n);
px=zeros(n,m);py=zeros(n,m);
%define vector component in x and y
%length of direction vectors
vsize=1.0;
for i=1:n
    for j=1:m
        if D(i,j)==1
            px(i,j)=vsize;py(i,j)=0;
        elseif D(i,j)==2
            px(i,j)=0;py(i,j)=-vsize;
        elseif D(i,j)==3
            px(i,j)=-vsize;py(i,j)=0;
        elseif D(i,j)==4
            px(i,j)=0;py(i,j)=vsize;
        else
            px(i,j)=0;py(i,j)=0;
        end
    end
end
%change row component to to be plotted
py1=zeros(n,m);px1=zeros(n,m);
for i=1:n
    px1(i,:)=px(n-i+1,:);
end
for i=1:n
    py1(i,:)=py(n-i+1,:);
end
figure('Position',[50 100 sz1 sz2],'name','A Network','NumberTitle','off','GraphicsSmoothing','off');

h=quiver(x,y,px1,py1,'k');
%set(h,'LineWidth',1.0);
set(h,'ShowArrowHead','off');
% set(h,'ShowArrowHead','on');
set(h,'AutoScaleFactor',0.8);
% set(h,'AutoScaleFactor',1.0);
% set(findobj('type','axes'),'linewidth',1.5);
h.AlignVertexCenters = 'on';

set(findobj('type','axes'),'visible','off');
hold on

%find values of FD in order and make an array
OA=sort(reshape(wts,n*m,1)); %ordered array
OAM=zeros(length(OA),1); %ordered array modified
OAM(1)=OA(1);
for j=2:length(OA)
    for i=1:length(OA)
        if OA(i)>OAM(j-1)
            OAM(j)=OA(i);
            break
        end
        if OAM(j)==max(OA)
            i_max=j;
            break
        end
    end
    if OAM(j)==max(OA)
        i_max=j;
        break
    end
end
OAM=OAM(1:i_max);


for k=1:length(OAM)
    for i=1:n
        for j=1:m
            if wts(i,j)<=OAM(k)
                D(i,j)=0;
            end
        end
    end

    %length of direction vectors
    vsize=1.0;
    for i=1:n
        for j=1:m
            if D(i,j)==1
                px(i,j)=vsize;py(i,j)=0;
            elseif D(i,j)==2
                px(i,j)=0;py(i,j)=-vsize;
            elseif D(i,j)==3
                px(i,j)=-vsize;py(i,j)=0;
            elseif D(i,j)==4
                px(i,j)=0;py(i,j)=vsize;
            else
                px(i,j)=0;py(i,j)=0;
            end
        end
    end
    %change row component to to be plotted
    py1=zeros(n,m);px1=zeros(n,m);
    for i=1:n
        px1(i,:)=px(n-i+1,:);
    end
    for i=1:n
        py1(i,:)=py(n-i+1,:);
    end


    % overlap quiver
    % first quiver with line width 0.1
    h=quiver(x,y,px1,py1,'k');
    set(h,'LineWidth',1.0+OAM(k)/max(OAM)*5.0);
    set(h,'ShowArrowHead','off');
    h.AlignVertexCenters = 'on';

    %set(h,'AutoScaleFactor',0.8);
    hold on
end
% tt=strcat('\bf\itA Network Simulated (',model_title,')');
% title(tt,'Color','k')

%matrix to show
if show==1
    VP=wts; % <--define here what to show
    for i=1:n
        for j=1:m
            if VP(i,j)~=0
                text(j+0.20,n-i+1+0.30,num2str(VP(i,j)))
            end
        end
    end
end
xlim([0 m+1]);ylim([0 n+1]);