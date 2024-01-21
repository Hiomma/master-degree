function [varargout] = ncpComp(S)
%[tr,VRE] = ncpComp(S) retorna o trace e o VRE para escolha do número de
%componentes principais, esse algoritmo é puxado na função indexComb.
%ncpComp(S) retorna o plot da variância acumulada e do VRE
%S = matriz de covariância

%Decomposição de autovalores
% b = size(S);
% [V,D] = eigs(S,b(1),'sm');
[V1,D1] = eig(S);
[V,D] = sortem(V1,D1);


SIGMA = V*(D)*V';
csi = eye(size(D,2));
for j = 1:size(D,2) %variavel
    for l = 1:size(D,2) %cp
% Matriz de loads e lodas residuais
Pchapeu = V(:,1:l);
Ptil = V(:,l+1:end);
C_til = Ptil*(Ptil)';
csitil = csi*C_til;
        sigma(j,l) = (csitil(:,j)'*SIGMA*csitil(:,j))/(csitil(:,j)'*csitil(:,j))^2;
    end
end


VRE = zeros(size(D,2),1);
for l = 1:size(D,2)
    for j = 1:size(D,2)
        VRE(l) = VRE(l) + sigma(j,l)/(csi(:,j)'*SIGMA*csi(:,j));
    end
end

for i = 1:size(D,2)
    tr(i,1) = 100*trace(D(1:i,1:i))/trace(D);
end

if nargout == 0
figure
ax1 = subplot(2,1,1);
plot(tr,'-d','MarkerSize',10)
title('Análise PCA (escolha do n\_cp)')
xlabel('n\_cp')
ylabel('% da variância')

ax2 = subplot(2,1,2);
plot(VRE,'-d','MarkerSize',10)
title('VRE')
xlabel('n\_cp')
ylabel('VRE')

linkaxes([ax1,ax2],'x')


elseif nargout == 2 
    varargout{1} = tr;
    varargout{2} = VRE;
    
else
    
    error('Número de saídas especificado de forma errada, usar [tr,VRE] = ncpComp(S)');

end
end