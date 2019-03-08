% check full connectivity of a direction matrix D

% Yongwon Seo
% 08/21/2018

tic

clear all
close all

load('gen_tr_layer1_labels_0.mat')

DD = zeros(11,11,10);
cnt = 0;
for i = 1:1000
    %eval(sprintf('D=D%04d;',i));
    D = t01(2:12,2:12,i);
    % matrix dimension
    n=length(D);m=n;
    % define outlet
    n0=n;m0=(m+1)/2;
    D(n0,m0)=2;

    % plot networks
    %plotdir_new(n,m,D,ones(n,m),0,256,256)
    %plotdir_new(n,m,D_d,ones(n,m),0,256,256)

    % use calculateq2 to calculate total flows of a direction matrix D
    [~,~,q]=calculateq2(D,ones(n,m),n,m,n0,m0);
    %[~,~,q_d]=calculateq2(D_d,ones(n,m),n,m,n0,m0);

    % if sum(q) is equal to the catchment area, D is fully connected and vice
    % versa

    % for D fully connected
    D1 = D;
    D(D~=0)=1;
    %disp('D:')
    if sum(q) == sum(sum(D))
        %disp('fully connected')
        fprintf('fully connectd: %d\n',i);
        cnt = cnt + 1;
        DD(:,:,cnt) = D1;
        plotdir_new(n,m,D1,ones(n,m),0,256,256)
    else
        %disp('not fully connected')
    end
    clear D D1
end

fprintf('fully connected %d\n', cnt);


%figure('Position',[50 100 256*2+50 256]','NumberTitle','off','GraphicsSmoothing','off');
%subplot(1,2,1)
%plot_publication(n,m,DD(:,:,7),ones(n,m),0)
%subplot(1,2,2)
%plot_publication(n,m,DD(:,:,9),ones(n,m),0)
