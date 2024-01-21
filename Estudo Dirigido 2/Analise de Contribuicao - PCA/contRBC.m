function [RBC_T2_perc, RBC_SPE_perc, RBC_phi_perc] = contRBC(Dindex,Ctil,PHI,X)

csi = eye(size(X,2));

%Contribuição RBC
for i = 1:size(X,2)
    for d=1:size(X,1)
        RBC_T2(d,i) = X(d,:)*Dindex*csi(:,i)*inv(csi(:,i)'*Dindex*csi(:,i))*csi(:,i)'*Dindex*X(d,:)';
        RBC_SPE(d,i) = X(d,:)*Ctil*csi(:,i)*inv(csi(:,i)'*Ctil*csi(:,i))*csi(:,i)'*Ctil*X(d,:)';
        RBC_phi(d,i) = X(d,:)*PHI*csi(:,i)*inv(csi(:,i)'*PHI*csi(:,i))*csi(:,i)'*PHI*X(d,:)';
    end
end
    
    %Contribuição RBC percentual
    for f = 1:size(RBC_phi,1)
        RBC_T2_perc(f,:) = RBC_T2(f,:)/sum(RBC_T2(f,:));
        RBC_SPE_perc(f,:) = RBC_SPE(f,:)/sum(RBC_SPE(f,:));
        RBC_phi_perc(f,:) = RBC_phi(f,:)/sum(RBC_phi(f,:));
    end
end