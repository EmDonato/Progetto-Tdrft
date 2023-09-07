function [] = CreateGraph(w,I,soglia)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
  
TP = 0;
TN = 0;
FP = 0;
FN = 0;


d = size(w);
ausI = I;
M = d(1);
N = d(2);
for j = 1:N
  for i = 1:M
        if abs(ausI(i,j)) > 0
             ausI(i,j)  = 1;
        end
   end
end
ausIcolomn = ausI(:);
ausWcolomn = w(:);
ausWcolomnCheck = ausWcolomn;
%Roc

numTrials = 10000;
if max(ausI) == 0
    maxnumb = 1;
    
else
     maxnumb = max(ausWcolomn);
end

threshold = linspace(0,maxnumb,numTrials);
disp('threshold');
disp(threshold);
TPR = zeros(numTrials,1);
FPR = zeros(numTrials,1);
dn = size(ausIcolomn);

for j = 1:N
  for i = 1:M
        if abs(ausI(i,j)) > 0
             ausI(i,j)  = 1;
        end
   end
end
ausIcolomn = ausI(:);
ausWcolomn = w(:);
ausWcolomnCheck = ausWcolomn;


 for i = 1 :dn(1)

    if ausWcolomnCheck(i,1) >= soglia
         ausWcolomn(i,1) = 1;
    else
         ausWcolomn(i,1) = 0;
    end

    if ausIcolomn(i,1) - ausWcolomn(i,1) == 0 && ausIcolomn(i,1) == 1
       TP = TP + 1; 
    elseif ausIcolomn(i,1) -ausWcolomn(i,1) == 1  
       FN = FN +1;
    elseif ausIcolomn(i,1) - ausWcolomn(i,1) == -1
       FP = FP +1;
    elseif ausIcolomn(i,1) - ausWcolomn(i,1) == 0 && ausIcolomn(i,1) == 0
       TN = TN + 1;

    end

 end



%% tabella di contingenza
figure(67);
cm=confusionchart(ausIcolomn,ausWcolomn);
cm.Title = 'tabella di contingenza soglia migliore';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';


accuratezza = (TP + TN) / (TP + TN + FP + FN);
precisione = TP / (TP + FP);
richiamo = TP / (TP + FN);
F1_score = 2 * (precisione * richiamo) / (precisione + richiamo);

%disp(['con la soglia migliore: ', num2str(soglia_migliore)]);


% Calcola l'Area Under the Curve (AUC) utilizzando la funzione trapz
AUC = trapz(FPR, TPR);

% Visualizza l'AUC calcolata
fprintf('Area Under the Curve (AUC): %.2f\n', abs(AUC));
disp(['Accuratezza: ', num2str(accuratezza)]);
disp(['Precisione: ', num2str(precisione)]);
disp(['Richiamo: ', num2str(richiamo)]);
disp(['F1-score: ', num2str(F1_score)]);


W_graph = reshape(ausWcolomn, M, N);
% Ottieni le coordinate degli archi nella matrice
[righe, colonne] = find(W_graph);

% Crea il grafo orientato
grafo = digraph(righe, colonne);

% Visualizza il grafo
figure(99)
plot(grafo);



end

