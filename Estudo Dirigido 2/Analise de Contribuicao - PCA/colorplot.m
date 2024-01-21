function colorplot(X)
%Entra com a matriz que se quer plotar, as subdivisões padrão são: 0-5, 5.1-10, 10.1-20, 20.1-40, 60.1-80, 80.1-100%
%Valores das divisões retirados do artigo Liu/Chen

% X = [X zeros(size(X,1),1)];
% S = pcolor(100*X');
S = imagesc(100*X');
C = get(S,'CData');
map = [1 1 1;[193 75 5] ./ 255;[249 7 240] ./ 255;;1 0 0;[139 1 6] ./ 255;0 0 0];
colormap(map);
C2 = zeros(size(C));
C2(C<5) = 1;
C2(C>=5 & C<10) = 2;
C2(C>=10 & C<20) = 3;
C2(C>=20 & C<40) = 4;
C2(C>=40 & C<80) = 5;
C2(C>=80) = 6;
set(S,'CData',C2,'CDataMapping','direct');
c = colorbar('Ticks',[1,2,3,4,5,6],'TickLabels',{' ' ,'5%','10%','20%','40%','80%'},'Location','eastoutside');
% set(c,'position',[0.92 0.725 0.03 0.2]);
% axis ij
% axis square
set(gca,'YDir','normal')
set(c,'position',[0.908 0.725 0.03 0.2]);
xlabel('Tempo após falha')
ylabel('Variável')


hold on
%Linha vertical
for i = 1:size(X',2)
    plot([i+0.5,i+0.5],[0 size(X',1)+1],'k','LineWidth',0.25)
   
end
%Linha horizontal
for i = 1:size(X',1)
    plot([0:size(X,1)+1],(i+0.5)*ones(size(X,1)+2),'k','LineWidth',0.25)
end
% xlim([1 size(X,1)])

end