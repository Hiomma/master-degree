function RCI_perc = contRCI(fault,no_fault,PHI,X)
%Cálculo do RCI segundo o artigo Liu/Chen
%Entrada: Variáveis que contribuiram pra falha; Variáveis que não contribuiram pra falha, matriz do index combinado, dados normalizados
csi = eye(size(X,2));

Tau = eye(size(X,2));
for i = 1:size(no_fault,2)
    Tau(no_fault(i),no_fault(i)) = 0;
end

xnf = zeros(size(X,1),size(X,2));
xnf(:,fault) = X(:,fault);

for j = 1:size(X,1)
    xnf_rec(j,:) = -inv(csi'*PHI*csi)*csi'*PHI*(eye(size(Tau,2))-Tau)*X(j,:)';
end

for i = 1:size(X,2)
    c_RCI(:,i) = ((xnf - xnf_rec)*((csi'*PHI*csi)^0.5)*csi(:,i)).^2;
end

for f = 1:size(c_RCI,1) 
RCI_perc(f,:) = c_RCI(f,:)/sum(c_RCI(f,:));
end
end
