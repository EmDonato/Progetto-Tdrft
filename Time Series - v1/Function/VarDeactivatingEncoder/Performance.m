function [TP,TN,FP,FN,W_graph] = Performance(w,I)
%Fa molte cose purtroppo, calcola i falsi e veri positivi/negativi,
% con la curva di Roc determina la migliore soglia 
%restituisce la matrice della causaliat con cui fare il grafo
  

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


% Calcolo del valore minimo e massimo del vettore
min_val = min(ausWcolomn);
max_val = max(ausWcolomn);

% Normalizzazione Min-Max
ausWcolomn_normalizzato = (ausWcolomn - min_val) / (max_val - min_val);

% curva di ROC
[X,Y,T,AUC,OPTROCPT] = perfcurve(ausIcolomn, ausWcolomn_normalizzato, 1);


% Visualizza la curva ROC
figure(42);
plot(X,Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Curva ROC');

% Visualizza l'AUC
disp(['AUC: ', num2str(AUC)]);


% Seleziona la soglia corrispondente all'indice
soglia_migliore = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)));

disp(['Soglia migliore: ', num2str(soglia_migliore)]);


%booleanizzo i risultati

TP = 0;
TN = 0;
FP = 0;
FN = 0;
d = size(ausIcolomn);

for i = 1:d(1)

    if ausWcolomn_normalizzato(i,1) >= soglia_migliore
         ausWcolomn_normalizzato(i,1) = 1;
    else
         ausWcolomn_normalizzato(i,1) = 0;
    end
    if ausIcolomn(i,1) - ausWcolomn_normalizzato(i,1) == 0 && ausIcolomn(i,1) == 1
       TP = TP + 1; 
    elseif ausIcolomn(i,1) -ausWcolomn_normalizzato(i,1) == 1  
       FN = FN +1;
    elseif ausIcolomn(i,1) - ausWcolomn_normalizzato(i,1) == -1
       FP = FP +1;
    elseif ausIcolomn(i,1) - ausWcolomn_normalizzato(i,1) == 0 && ausIcolomn(i,1) == 0
       TN = TN + 1;

    end 

end

figure(43);
%% tabella di contingenza

cm=confusionchart(ausIcolomn,ausWcolomn_normalizzato);
cm.Title = 'tabella di contingenza soglia migliore';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';


accuratezza = (TP + TN) / (TP + TN + FP + FN);
precisione = TP / (TP + FP);
richiamo = TP / (TP + FN);
F1_score = 2 * (precisione * richiamo) / (precisione + richiamo);

disp(['con la soglia migliore: ', num2str(soglia_migliore)]);

disp(['Accuratezza: ', num2str(accuratezza)]);
disp(['Precisione: ', num2str(precisione)]);
disp(['Richiamo: ', num2str(richiamo)]);
disp(['F1-score: ', num2str(F1_score)]);




%% ritrasforma in matrice la w

    W_graph = reshape(ausWcolomn_normalizzato, M, N);
    

end
