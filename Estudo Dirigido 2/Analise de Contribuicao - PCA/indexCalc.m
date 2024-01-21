function [T2,SPE,phi] = indexCalc(Dindex,Ctil,PHI,X)

for i = 1:size(X,1)
    SPE(i,1) = X(i,:)*Ctil*X(i,:)';
    T2(i,1) = X(i,:)*Dindex*X(i,:)';
    phi(i,1) = X(i,:)*PHI*X(i,:)';
end
end