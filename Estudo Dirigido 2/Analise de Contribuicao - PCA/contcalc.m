function [C_T2_perc, C_SPE_perc, C_phi_perc] = contcalc(Dindex,Ctil,PHI,X)
%Contribuição em T2, SPE e index combinado
%Entradas, Matrizes características de T2, SPE e phi e dados normalizados

csi = eye(size(X,2));

for i = 1:size(X,2)
for d=1:size(X,1)
    C_T2(d,i) = real((csi(:,i)'*(Dindex^0.5)*X(d,:)')^2);
    C_SPE(d,i) = real((csi(:,i)'*(Ctil^0.5)*X(d,:)')^2);
    C_phi(d,i) = real((csi(:,i)'*(PHI^0.5)*X(d,:)')^2);
end
end

%Contribuições em percentual
for f = 1:size(C_phi,1)
    C_T2_perc(f,:) = C_T2(f,:)/sum(C_T2(f,:));
    C_SPE_perc(f,:) = C_SPE(f,:)/sum(C_SPE(f,:));
    C_phi_perc(f,:) = C_phi(f,:)/sum(C_phi(f,:));
end
end