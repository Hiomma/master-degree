function [a,limT2,limQ,limphi,Dindex,Ctil,phi]=indexComb(S,X,alpha)
% Função para cálculo das estatísticas t2 e Q.
% Também retorna o limiar de T2 e o número de componentes principais (a) que
% contém 99% da variância dos dados.
% Os argumentos de entrada são a matriz de covariância e os dados.



%Decomposição de autovalores
% b = size(S);
% [V,D] = eigs(S,b(1),'sm');
[V1,D1] = eig(S);
[V,D] = sortem(V1,D1);


%Calculo das componentes principais
% cumvar = cumsum(diag(D));
% cvar = cumvar/cumvar(end);
% cp = find(cvar>0.90);
% a=cp(1);

[cuvar, vre] = ncpComp(S);
cp = find(cuvar>90);
[~,m] = min(vre(cp));
a = m + (length(cuvar)-length(cp));

P = V(:,1:a);
T = X*P;
lambda = D(1:a,1:a);
Dinv = inv(lambda);
% n = size(X_norm,1);

% %Calculo da estatistica Q
% E = X - T*P';
% Q = diag(E*E');

% %Calculo da estatistica T2
% T2 = zeros(1,length(X));
% for i=1:length(T)
% T2(i) = T(i,:)*Dinv*T(i,:)';
% end
% n = length(T2);
n = size(X,1);

%Limiares de T2 e Q
limT2 = (a*(n-1)/(n-a))*finv(alpha,a,n-a);
[limQ,teta1,teta2]=qstat(S,alpha,a);

%Indice Combinado entre T2 e Q
%Calculo das matrizes M, baseado em Alcala, Qin, 2006
Ptil = V(:,a+1:end);
Ctil = Ptil*Ptil';
Dindex = P*Dinv*P';

phi = Ctil/(limQ) + Dindex/(limT2);
gphi = ((a/limT2^2)+(teta2/limQ^2))/((a/limT2)+(teta1/limQ));
hphi = ((a/limT2)+(teta2/limQ))/((a/limT2^2)+(teta2/limQ^2));
limphi = gphi*chi2inv(alpha,hphi);

%phimin = Q/limQ + T2'/limT2;
end



