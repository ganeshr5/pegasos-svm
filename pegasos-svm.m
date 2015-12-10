% Pegasos Implementation
% MNIST-13 Dataset

function [tm] = pegasos-svm(filename, k, numruns)
% filename - data
% k - batch size
% numruns - number of runs

clear all
clc
clf
close all

filename='MNIST-13.csv';
warning('OFF');
dat = csvread(filename);
n = size(dat,1);
ndat = dat(:, 2:end);  % new data (class labels removed)
cl = dat(:,1);     % class labels
for i = 1:n
    if (cl(i) == 3)
        cl(i) = -1;
    end
end

X = ndat;
Y = cl;
lamda=1; 
k=1;
Tolerance = 10^-2;
wq=0; mi=0; bm=0;
tic;
for j= 1:5
    wT = 0;
    maxIter=0;
    maxIter = j*100;  
    [N,d]=size(X);
    w=rand(1,size(X,2));
    w=w/(sqrt(lamda)*norm(w));
    for t=1:maxIter
        b=mean(Y-X*w(t,:)');
        idx=randint(k,1,[1,size(X,1)]);
        At=X(idx,:);
        yt=Y(idx,:);
        idx1=(At*w(t,:)'+b).*yt<1;
        etat=1/(lamda*t);
         % update w
        w1=(1-etat*lamda)*w(t,:)+(etat/k)*sum(At(idx1,:).*repmat(yt(idx1,:),1,size(At,2)),1);
        % projection of w 
        w(t+1,:)=min(1,1/(sqrt(lamda)*norm(w1)))*w1; 
        mw(t+1)=mean(w(t+1,:));
        plot(t, mean(mw));
        hold on;
        title('k');
        ylabel('Primal value')
        xlabel('No of Iterations')
    end
    wT=mean(w,1);
   %wTstd = std(wT)
    % wT=w(end,:);
    b=mean(Y-X*wT');
    bm(i) = b; 
    mi(i)=maxIter;
end
wTstd = std(wT)
tm=toc;
fprintf('time = %f', tm);

end
