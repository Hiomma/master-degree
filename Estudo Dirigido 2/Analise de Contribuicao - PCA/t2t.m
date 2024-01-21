function M=t2t(X,p)
% function M=t2t(X,p)
% Input:
%   X = matrix n.m
%   p = [alfa limvar]  = [Confidence level   variance to keep  ou Number of PC]
%
% Output: Model M
% Data: 8/jun/2022
%
if nargin==1
    alfa=0.95;
    limvar=0.95;
else
    alfa=p(1);
    limvar=p(2);
end;
[n,m]=size(X);
mu=mean(X);
st=std(X);
X=bsxfun(@minus, X, mu); % Detrend X
X=bsxfun(@rdivide, X, st);
[~,S,v]=svd(cov(X));
sd=diag(S);
sst=sum(sd);
sd=sd/sst;
ss=sd(1);
a=1;
if (limvar>0.99)&(limvar<=1)
    a=m;
elseif limvar<=0.99
    while ss<limvar % Select principal components
        a=a+1;
        ss=ss+sd(a);
    end;
else
    a=limvar;
end
P=v(:,1:a);
r=X*(eye(m)-P*P');
M.alfa=alfa;
M.mu=mu;
M.st=st;
M.P=P; % Loading vectors
M.S=S; % Covariance matrix
M.a=a; % Principal components
M.n=n; % Number of samples used for trainig
if a<m
    M.r_var=var(r); % Variance of residue
else
    M.r_var=[];
end
end