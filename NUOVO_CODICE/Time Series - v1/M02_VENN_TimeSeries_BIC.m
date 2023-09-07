% Questa sezione serve solo a "Pulire" gpu, workspace, etc.
% 
% gpu = gpuDevice();
% reset(gpu)
% 
% gpu = gpuDevice();
% disp(gpu)
% wait(gpu)
% 
% clear all; clc; warning off;
% 
% figure(3)
% clf
%%
clear all; clc; warning off; close all;
%%
% parametri per i cicli

NTarget = 3;% cambiare 
W_old = zeros(NTarget);
Nepoch = 100;
epoch = 1
W_matrix = zeros(NTarget,NTarget);
i_target = 1;
save('W_old.mat','W_old');
save('W_matrix.mat','W_matrix');
save('i_target.mat','i_target');
%%
    autoregressiveST = ['Autoregressive_v2.mat',"Autoregressive_v3.mat",'Autoregressive_v4.mat','Autoregressive_v5.mat','Autoregressive_v6.mat','Autoregressive_v7.mat'];

for j_autoregressive_targhet = 1:6
    save('j_autoregressive_targhet.mat','j_autoregressive_targhet');
    i_target = 1;
    save('i_target.mat','i_target');
    fprintf('Valore di j_autoregressive_targhet: %.4f\n', j_autoregressive_targhet);
    Nepoch = 100;
    epoch = 1;
for i_target = 1 : NTarget

    W_old = struct2array(load('W_old.mat'));
    clear all; clc; warning off; close all;
    autoregressiveST = ['Autoregressive_v2.mat',"Autoregressive_v3.mat",'Autoregressive_v4.mat','Autoregressive_v5.mat','Autoregressive_v6.mat','Autoregressive_v7.mat'];
    epoch = 1
    % parametri per i cicli
    Nepoch = 100;
    NTarget = 3; % cambiare
    W_matrix = struct2array(load('W_matrix.mat'));
    i_target = int32(struct2array(load('i_target.mat')));
    j_autoregressive_targhet = int32(struct2array(load('j_autoregressive_targhet.mat')));

    figure(3)
    clf



%% load data 
% (qua decidete quale dataset volete analizzare,
% potete generarli a piacere)
addpath("Datasets\Synthetic\")
load(autoregressiveST(j_autoregressive_targhet))






%% add paths (dove ci sono le funzioni utilizzate dopo)

addpath("Function\VarDeactivatingEncoder\")

%% Prepare input and output

% extract output
Y = X(i_target,:);

% select delays (per ora lasciate così, poi vi spiego)
delays = [1];

% Da qua fino alla prossima selezione mi giro i dati in una maniera comoda
% (la prossima volta vi spiego bene)
maximum_delays = max(delays);

D = [];

Label_features = [];

for i = 1 : size(X,1)

    Label_features = [Label_features; "x_"+ i + "(t - " + delays' + ")"];
    
    B = buffer(X(i,:),maximum_delays,maximum_delays-1);

    D = [D; B(delays,:)];

end

Features = D;

Features = Features(:,maximum_delays:end-maximum_delays)';
Targets = Y(2*maximum_delays:end)';

%% weights generation
% (lasciate zero, i pesi saranno tutti uguali a 1 
% e quindi inutili)

MSE_weights = 0;

if MSE_weights == 1

    DeltaY = (max(Y) - min(Y))/floor(length(Y)^0.5);
    YDensity = sum((abs(Y-Y'))<DeltaY,2);
    W_density = (YDensity);

else

    W_density = ones(size(Y));

end

%% Normalisation (normalizzo i dati)

Norm.Fmean = mean(Features,1);
Norm.Fstd = std(Features,[],1);

Norm.Tmean = mean(Targets,1);
Norm.Tstd = std(Targets,[],1);

Features = (Features' - Norm.Fmean')./Norm.Fstd';
Targets = (Targets' - Norm.Tmean')./Norm.Tstd';

%% dlarray (prepare data for training and prediction)

Train.minibatchsize = 1000;

TrainingSet = 0.9;
NTrainingSet = floor(TrainingSet*length(Y));
ind = 1:NTrainingSet;

Train.IterationPerEpoch = floor(NTrainingSet/Train.minibatchsize);

Train.ind_inputs = 1:size(Features,1);
Train.ind_outputs = (1:size(Targets,1)) + Train.ind_inputs(end);

ds = arrayDatastore([Features(:,ind)',Targets(:,ind)',W_density(:,ind)']);

mbq = minibatchqueue(ds,...
    "MiniBatchSize",Train.minibatchsize,...
    "MiniBatchFormat","BC",...
    "OutputEnvironment","auto");

dX = next(mbq);

Xtest = Features';
Xtest(ind,:) = [];
Ytest = Targets';
Ytest(ind,:) = [];

dXtest = dlarray(Xtest,'BC');

%% Generate Network

% RF è un parametro libero che non ha una grande influenza, lasciare così
RF = 0.1;

% genero la rete neurale (inizializzo la struttura parameters dove ci sono
% tutti i parametri della rete)
[Yp,parameters] = VarEnc_deAct(dX(1:size(Features,1),1),dX(size(Features,1)+1,1),0,[],RF);

% Testo la rete (qua la rete non è addestrata, serve solo per vedere se
% funziona)
Yp = VarEnc_deAct(dX(1:size(Features,1),:),dX(size(Features,1)+1,:),1,parameters,RF);

reset(mbq)

clear Yp

%% Training options

% parametro per cambiare alpha in maniera automatica, ne parliamo la
% prossima volta
Z_threshold = 2;

% inizializzo alcuni parametri per il training
iteration = 0;

averageGrad = [];
averageSqGrad = [];

% parametri per gestire il learning rate
L0 = 1e-2;
D0 = 1e-3;

%% Initialise Best

BIC_Best = 1e10;
validation_checks = 0;
validation_max = 100;

%% Model Gradient
% Questa è la funzione che calcola la loss e il gradiente, verrà richiamata
% ad ogni iterazione per addestrare la rete
accfun = dlaccelerate(@ModelGradient_VarEnc_deActivation_BIC);

%% Training
% da qua inizia il processo di training vero e proprio

for epoch = 1 : Nepoch

    % resetto e randomizzo i dati di training
    reset(mbq)
    shuffle(mbq)

%display epoche
% disp('epoch = ');
% disp(epoch);
% disp('Nepoch = ');
% disp(Nepoch);
    % svolgo un'iterazione per ogni minibatch
    for i = 1 : Train.IterationPerEpoch

        iteration = iteration + 1;

        dX = next(mbq);

        % Output
        dY = dX(size(Features,1)+1,:);
        % Pesi (inutili per ora)
        dW = dX(size(Features,1)+2,:);
        % Inputs
        dX = dX(1:size(Features,1),:);

        % Richiamo il model gradient, calcolo la loss e i gradienti        
        [gradients,Loss,MSE,K] = dlfeval(accfun,parameters,dX,dY,RF,dW);

        % Update learning rate.
        learningRate = max(L0./(1 + D0*iteration),1e-4);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);

    end

    %% Predict train and test
    % qua predico gli output e calcolo la loss nell'ultimo batch di
    % training e nel test set. Questi dati verrano utilizzti sia per fare i
    % plot che seguono sia per le varie condizioni di stop

    % Predizione sul training
    [dYp,parameters] = VarEnc_deAct(dX,dY(:,1),1,parameters,0);

    % Per il test faccio un campionamento randomico se il test set è
    % maggiore di N = 5e3 (per evitare un sovraccarico durante il test, che
    % sarebbe inutile)
    ktest = randsample(length(Ytest),min(length(Ytest),5e3));

    % Predizione test
    [dYptest,parameters] = VarEnc_deAct(dXtest(:,ktest),dY(:,1),1,parameters,0);
    Yptest = double(extractdata(gather(dYptest)))';

    % calcolo le loss nei due casi
    MSE_Test = mean((Ytest(ktest,:)-Yptest).^2);


    % Calcola falsi positivi e falsi negativi
  %  false_positives = sum(predictions == 1 & labels == 0);
  %  false_negatives = sum(predictions == 0 & labels == 1);

    % questi sono i pesi del primo layer per fare feature selection
    W = double(extractdata(gather(parameters.W.weights)));
    K = sum(abs(W/RF),'all');

    BIC_test = length(ktest)*log(MSE_Test) + K.*log(length(ktest));
    
    %% Faccio vari plot

    figure(2)
    subplot(2,3,1)
    hold on
    plot(epoch,Loss,'.b','markersize',16)
    plot(epoch,BIC_test,'.r','markersize',16)
    set(gca,"YScale",'log')
    grid on
    grid minor
    xlabel("epoch")
    ylabel("BIC")

    subplot(2,3,2)
    hold on
    plot(K,MSE_Test,'.b')
    set(gca,"YScale",'log')
    grid on
    grid minor
    xlabel("Determenism")
    ylabel("MSE")
    
    subplot(2,3,3)
    hold off
    plot(dY,dYp,'.b','markersize',16)
    hold on
    plot(Ytest(ktest,:),Yptest,'.r','markersize',12)
    plot([-5 5],[-5 5],'-.k','LineWidth',2)
    xlabel("Target")
    ylabel("Predicted")
    legend("Training","Test")
    grid on
    grid minor

    subplot(2,3,[4 5 6])
    hold off
    plot(1:length(W),ones(size(W)),'-.k','linewidth',2)
    hold on
    plot(1:length(W),ones(size(W))*2,'--r','linewidth',2)
    plot(1:length(W),abs(W./RF),'ob','linewidth',2)
    ylim([0 inf])
    grid on
    grid minor
    xticks(1:length(W))
    ylabel("VES")

    drawnow
    
    %% Best

%     if BIC_test < BIC_Best
%         
%         Best.epoch = epoch;
%         Best.parameters = parameters;
%         Best.BIC = BIC_test;
%         Best.W = W;
%         Best.RF = RF;
% 
%         BIC_Best = BIC_test;
% 
%         validation_checks = 0;
% 
%     else
% 
%         validation_checks = validation_checks + 1;
% 
%         disp(validation_checks)
%         
%         if validation_checks >= validation_max
% 
%             break
% 
%         end
% 
%     end
  W_old = struct2array(load('W_old.mat'));
%     if(all(abs(abs(W_old)-abs(W))<0.001))
%         fprintf('USCITOOOOOOOO');
%         break
%     end
    save('W_old.mat','W');

     W_display = abs(abs(W_old)-abs(W));
     %disp(W_display);

end





switch i_target
    case 1
        save('w1.mat', 'W');
    case 2
        save('w2.mat', 'W');
    case 3
        save('w3.mat', 'W');
    case 4
        save('w4.mat', 'W');
    case 5
        save('w5.mat', 'W');
    case 6
        save('w6.mat', 'W');        
    otherwise
        disp("errore servono piu variabili");
end
W_matrix(:,i_target) = W;
i_target = i_target + 1;
save('j_autoregressive_targhet.mat','j_autoregressive_targhet');

%fprintf('Il valore di i_target è %.2f\n', i_target);
save('i_target.mat','i_target');
%W_matrix(W_matrix < 0.1) = 0;
%W_matrix(W_matrix > 0.1) = 1;
save('W_matrix.mat','W_matrix');

% Best.W = Best.W';
% save("Last_Results","Best")
end
%%
switch j_autoregressive_targhet
    case 1
        save('I2.mat','I')
        save('W_matrix2.mat','W_matrix');
    case 2
        save('I3.mat','I')
        save('W_matrix3.mat','W_matrix');
    case 3
        save('I4.mat','I')
        save('W_matrix4.mat','W_matrix');
    case 4
        save('I5.mat','I')
        save('W_matrix5.mat','W_matrix');
    case 5
        save('I6.mat','I')
        save('W_matrix6.mat','W_matrix');
    case 6
        save('I7.mat','I')
        save('W_matrix7.mat','W_matrix');
    otherwise
        disp("errore servono piu variabili");
end


j_autoregressive_targhet = j_autoregressive_targhet +1;
save('j_autoregressive_targhet.mat','j_autoregressive_targhet');




end

%% solo se sono 7 le matrici


  W_matrix2 = struct2array(load('W_matrix2.mat'));
  ausW2colomn = W_matrix2(:);
  W_matrix3 = struct2array(load('W_matrix3.mat'));
  ausW3colomn = W_matrix3(:);
  W_matrix4 = struct2array(load('W_matrix4.mat'));
  ausW4colomn = W_matrix4(:);
  W_matrix5 = struct2array(load('W_matrix5.mat'));
  ausW5colomn = W_matrix5(:);
  W_matrix6 = struct2array(load('W_matrix6.mat'));
  ausW6colomn = W_matrix6(:);
  W_matrix7 = struct2array(load('W_matrix7.mat'));
  ausW7colomn = W_matrix7(:);
  W_matrixTotal = horzcat(ausW2colomn,ausW3colomn,ausW4colomn,ausW5colomn,ausW6colomn,ausW7colomn);
  W_matrixTotalColomn = W_matrixTotal(:);
%% stessa cosa per le matrici I
  I2 = struct2array(load('I2.mat'));
  I2colomn = I2(:);
  I3 = struct2array(load('I3.mat'));
  I3colomn = I3(:);
  I4 = struct2array(load('I4.mat'));
  I4colomn = I4(:);
  I5 = struct2array(load('I5.mat'));
  I5colomn = I5(:);
  I6 = struct2array(load('I6.mat'));
  I6colomn = I6(:);
  I7 = struct2array(load('I7.mat'));
  I7colomn = I7(:);
  ITotal = horzcat(I2colomn,I3colomn,I4colomn,I5colomn,I6colomn,I7colomn);
  ITotalColomn = ITotal(:);

%%




%inizializzazione parametri per il calcolo delle performance
%[TrueP,TrueN,FalseP,FalseN,W_graph] = PerformanceNew(W_matrixTotal,ITotal)

[soglia,AUC] = myROC(W_matrixTotal,ITotal);

%%
CreateGraph(W_matrix7,I7,soglia)




