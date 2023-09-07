function [TP,TN,FP,FN,W_graph] = PerformanceNew(w,I)
%Fa molte cose purtroppo, calcola i falsi e veri positivi/negativi,
% con la curva di Roc determina la migliore soglia 
%restituisce la matrice della causaliat con cui fare il grafo
  
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

for j = 1:numTrials

TP = 0;
TN = 0;
FP = 0;
FN = 0;

 for i = 1 :dn(1)

     fprintf('Valore di threshold(j): %.4f\n', threshold(j));

    if ausWcolomnCheck(i,1) >= threshold(j)
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
 if TP == 0 && (TP+FN) == 0
     TPR(j) = 1;
     TPR(end) = 0;
 else
     TPR(j) =(TP/(TP+FN)) ;
 end
  if FP == 0 && (FP+TN) == 0
     FPR(j) = 0;
     FPR(1) = 1;
 else
     FPR(j) = (FP/(FP+TN)) ;
 end

fprintf('Valore di TP: %.4f\n', TP);
fprintf('Valore di FP: %.4f\n', FP);
fprintf('Valore di TN: %.4f\n', TN);
fprintf('Valore di FN: %.4f\n', FN);


fprintf('Valore di TPR: %.4f\n', TPR(i));
fprintf('Valore di FPR: %.4f\n', FPR(i));

end
 % ROC
figure(6)
plot(FPR, TPR, 'LineWidth', 2, 'MarkerSize', 8);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('Curva ROC');
grid on;


% Calcola l'Area Under the Curve (AUC) utilizzando la funzione trapz
AUC = trapz(FPR, TPR);

% Calcola la distanza di ciascun punto dalla bisettrice
distanze = abs(TPR - FPR);

% Trova l'indice del valore massimo di distanza
indiceMigliore = find(distanze == max(distanze));

disp('Indice del valore più lontano dalla bisettrice:');
disp(indiceMigliore(1));

disp(' valore del threshold migliore');
disp(threshold(indiceMigliore(end)));

% Calcola la somma di TPR e (1 - FPR) per ogni punto sulla curva ROC
metricaYouden = TPR + (1 - FPR);

% Trova l'indice in cui la metrica Youden è massima
indiceYouden = find(metricaYouden == max(metricaYouden));

disp('Indice Youden (punto ottimale sulla curva ROC):');
disp(indiceYouden);

TP = 0;
TN = 0;
FP = 0;
FN = 0;

 for i = 1 :dn(1)

    if ausWcolomnCheck(i,1) >= threshold(indiceYouden(end))
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




% 
% %booleanizzo i risultati
% 

% d = size(ausIcolomn);
% 
% for i = 1:d(1)
% 
%     if ausWcolomn(i,1) >= soglia_migliore
%          ausWcolomn(i,1) = 1;
%     else
%          ausWcolomn(i,1) = 0;
%     end
%     if ausIcolomn(i,1) - ausWcolomn(i,1) == 0 && ausIcolomn(i,1) == 1
%        TP = TP + 1; 
%     elseif ausIcolomn(i,1) -ausWcolomn(i,1) == 1  
%        FN = FN +1;
%     elseif ausIcolomn(i,1) - ausWcolomn == -1
%        FP = FP +1;
%     elseif ausIcolomn(i,1) - ausWcolomn == 0 && ausIcolomn(i,1) == 0
%        TN = TN + 1;
% 
%     end 
% 
% end
% 
figure(42);
%% tabella di contingenza

cm=confusionchart(ausIcolomn,ausWcolomn);
cm.Title = 'tabella di contingenza soglia migliore';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';


accuratezza = (TP + TN) / (TP + TN + FP + FN);
precisione = TP / (TP + FP);
richiamo = TP / (TP + FN);
F1_score = 2 * (precisione * richiamo) / (precisione + richiamo);

%disp(['con la soglia migliore: ', num2str(soglia_migliore)]);



% Visualizza l'AUC calcolata
fprintf('Area Under the Curve (AUC): %.2f\n', abs(AUC));
disp(['Accuratezza: ', num2str(accuratezza)]);
disp(['Precisione: ', num2str(precisione)]);
disp(['Richiamo: ', num2str(richiamo)]);
disp(['F1-score: ', num2str(F1_score)]);
% 
% 
% 
% 
% %% ritrasforma in matrice la w
% 
     W_graph = reshape(ausWcolomn, M, N);
%     
% 
disp(['threshold: ', threshold]);

 end
