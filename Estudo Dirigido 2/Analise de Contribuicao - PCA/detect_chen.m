function [fault,no_fault,p_falha] = detect_chen(X,X_treino)
%Retorna os vetores com as variáveis que auxiliaram e não para a falha
%Entrada Dados normalizados, matriz de dados usada pra normalizá-los


vars1 = [2:size(X,2)];
fault = [];
no_fault=[];
for i = 1:size(X,2)
    
    X_aux = X;
    
    if i<=size(X,2)-2
        aux2 = 2;
    elseif i == size(X,2)-1
        aux2 = 1;
    else
        aux2 = 0;
    end
    
    X_aux(:,vars1)=[];
    
    X_norm = X_aux(1:size(X_treino,1),:);
    
    % Matriz de covariância
    S = X_norm'*X_norm/(size(X_norm,1)-1);
    
    % Função do Arthur
    [~,limT2,~,~,Dindex,~,~]=indexComb(S,X_aux,0.99);
    
    for n = 1:size(X_aux,1)
        T(n,1) = X_aux(n,:)*Dindex*X_aux(n,:)';
    end
    
    %     cont = 0;
    %     for j = 1:length(T)
    %         if T(j)>limT2
    %             cont = cont +1;
    %         end
    %     end
    %
    %     ult = 100*cont/length(T);
    %
    %     if ult>10
    %         fault = [fault i];
    %         vars1 = [fault i+aux2:size(X,2)];
    %
    %     else
    %         vars1 = [fault i+aux2:size(X,2)];
    %         no_fault = [no_fault i];
    %     end
    
    
    encontrou = 0;
    for j = 1:length(T)
        cont = 0;
        if T(j)>limT2
%             if j<11
%                 analise = T(1:j+10,:);
            if j>length(T)-10
                analise = T(j:end,:);
            else
                analise = T(j:j+10,:);
            end
            for w = 1:size(analise,1)
                if analise(w)>limT2
                    cont = cont +1;
                end
            end
            ult = 100*cont/length(analise);
            if ult>70
                fault = [fault i];
                vars1 = [fault i+aux2:size(X,2)];
                encontrou = 1;
                p_falha(i) = j;
                break
            end
        end
    end

    if encontrou == 0
        vars1 = [fault i+aux2:size(X,2)];
        no_fault = [no_fault i];
    end
    
    
    
end
    y = find(p_falha==0);
    p_falha(y) = [];
end
