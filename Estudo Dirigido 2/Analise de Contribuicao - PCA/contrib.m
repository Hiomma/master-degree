function  C = contrib(x,M,type)
% Contribution of variables in x to the fault
C=[];
[n,m]=size(x);
x=bsxfun(@minus, x, M.mu);
switch type
    case 'pca' % Using PCA
        for i=1:n
            C=[C;contrib_pca(x(i,:),M)];
        end
        
    case 'q'  % Using residue
        C=x*(eye(m)-M.P*M.P');
        C=bsxfun(@rdivide, C, M.r_var); % Standardization by variance
        C=abs(C);
        
    case 'dv'  % Deviation from variables
        C=abs(x);
end
if nargout==0
    for i=1:m
        subplot(m,1,i);plot(C(:,i));ss=strcat('Variavel->',num2str(i));title(ss);
    end
end
end
%
function [ ctr, c ] = contrib_pca(x, M)
% Contribution using PCA for each sample
T=x*M.P; 
[m,c]=size(M.P); % m = number of variables
ct2=zeros(1,m);
ctr=zeros(1,m);
idx=[];
P1=eye(m)-M.P*(M.P');
t2=threshold(M,'t2');
for j=1:c % c scores
    if (((T(j)/sqrt(M.S(j,j)))^2)>(1/M.a)*t2)
        idx=[idx j];  % scores that violate threshold
    end
end;
cont=[];
c=length(idx); % c scores selecionados (violou limiares)
if c>0 % If at least one score was violated
    for i=1:c      % computation for each score
        for j=1:m  % Contribution of m variables to score ti
            tn=idx(i);
            ti=T(tn);
            pij=M.P(j,tn);
            aux=(ti/M.S(tn,tn))*pij*x(j);
            if aux>0
                cont(i,j)=aux;
            else
                cont(i,j)=0;
            end
        end
    end
    if c>1 cont= sum(cont); end; % Add contribution of m variables 
    ctr= cont/sum(cont); 

end 

end