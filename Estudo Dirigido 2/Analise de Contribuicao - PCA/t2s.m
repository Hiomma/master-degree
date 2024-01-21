function Y=t2s(y,M, type, II)
% Monitoring signal y using model M and stat type 
% Celso Munaro
% Data:11-Jun-2022
if nargin==3
    II=length(y);
end
[n,m]=size(y);
y=bsxfun(@minus, y, M.mu);
y=bsxfun(@rdivide, y, M.st); % Detrend X
zf=0;resf=0;
D=M.P*inv(M.S(1:M.a,1:M.a))*(M.P');
C=eye(m)-M.P*(M.P');
switch type
    case 't2'
        F=D;
        limiar=threshold(M,'t2');
    case 'q'
        if M.a<m
        F=C;
        limiar=threshold(M,'q');
        else
            Y=[];
            disp('Estatistica Q não se aplica');
            return;
        end
    case 'c'
        if M.a<m
        limiar_t2=threshold(M,'t2');
        limiar_q=threshold(M,'q');
        F=D/limiar_t2+C/limiar_q;
        limiar=threshold(M,'c');
        else
            Y=[];
            disp('Estatistica combinada não se aplica');
            return;
        end           
    otherwise
        disp('Escolha uma das opcoes');
        beep;
        Y=[];
        return;
end
for i=1:n 
    yn=y(i,:);
    z(i,1)=yn*F*(yn'); % Computes stats
end
Y.z=z;
Y.thr=limiar;
%FP=100*sum(Y.z>Y.thr)/length(Y.z);
FP=100*sum(Y.z(1:II)>Y.thr)/II;
Y.FP=FP;
if nargout==0
    plot([z ones(size(z))*limiar]);
    %ss=sprintf('Monitoring with %s  FP (Rever) = %2.0f %c' ,type,FP,'%');
    %title(ss,'FontSize',18);
end;
end