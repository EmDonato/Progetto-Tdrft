% Questa sezione serve solo a "Pulire" gpu, workspace, etc.

% gpu = gpuDevice();
% reset(gpu)
% 
% gpu = gpuDevice();
% disp(gpu)
% wait(gpu)

clear all; clc; warning off;

figure(3)
clf
i_target = 1;
%% load data 
% (qua decidete quale dataset volete analizzare,
% potete generarli a piacere)

%% load data 
% (qua decidete quale dataset volete analizzare,
% potete generarli a piacere)
addpath("Datasets\Synthetic\")
load("Autoregressive_v1.mat")

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

%% Variational Deactivation Neural Network

% rimuovo i dati che sono Not a Number (NaN) se ci sono
ind = find(sum(isnan(X),2)>=1);

X(ind,:) = [];
Y(ind,:) = [];

% Normalizzo gli input e gli output
Norm.Xmean = mean(X);
Norm.Xstd = max(std(X),eps);
Norm.Ymean = mean(Y);
Norm.Ystd = max(std(Y),eps);

X = (X - Norm.Xmean)./Norm.Xstd;
Y = (Y - Norm.Ymean)./Norm.Ystd;

% Definisco la percentuale di training set 
TrainingSet = 0.9;

% Campiono randomicamente dal dataset il training set 
NTrainingSet = floor(TrainingSet*length(Y));
ind = randsample(length(Y),NTrainingSet);

% Questa è una struttura che serve a definire il
% "mini batch queue", un metodo per gestire velocemente i batch
ds = arrayDatastore([X(ind,:) Y(ind) W_density(ind)]);

minibatch = 1000;
IterationPerEpoch = floor(NTrainingSet/minibatch);

mbq = minibatchqueue(ds,...
    "MiniBatchSize",minibatch,...
    "MiniBatchFormat","BC",...
    "OutputEnvironment","auto");  %%CPU / GPU/ auto

% esempio per richiamare un batch
dX = next(mbq);

% Definisco il test set
Xtest = X;
Xtest(ind,:) = [];
Ytest = Y;
Ytest(ind) = [];

% Trasforme il test set in un dlarray (struttura dati usata per le reti in
% Matlab)
dXtest = dlarray(Xtest,'BC');

%% Generate Network

% RF è un parametro libero che non ha una grande influenza, lasciare così
RF = 0.1;

% genero la rete neurale (inizializzo la struttura parameters dove ci sono
% tutti i parametri della rete)
[Yp,parameters] = VarEnc_deAct(dX(1:size(X,2),1),dX(size(X,2)+1,1),0,[],RF);

% Testo la rete (qua la rete non è addestrata, serve solo per vedere se
% funziona)
Yp = VarEnc_deAct(dX(1:size(X,2),:),dX(size(X,2)+1,:),1,parameters,RF);

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
D0 = 1e-4;

% iperparametro di regolarizzazione
alpha = 0.01;

%% Initialise train and test losses

% inizializzazione del vettore delle ultime 10 loss (vedere dopo)
LastLTrains = ones(10,1)*std(Y);
LastLTests = ones(10,1)*std(Y);

%% Model Gradient
% Questa è la funzione che calcola la loss e il gradiente, verrà richiamata
% ad ogni iterazione per addestrare la rete
accfun = dlaccelerate(@ModelGradient_VarEnc_deActivation);

%% Training
% da qua inizia il processo di training vero e proprio

for epoch = 1 : 3000

    % resetto e randomizzo i dati di training
    reset(mbq)
    shuffle(mbq)

    % svolgo un'iterazione per ogni minibatch
    for i = 1 : IterationPerEpoch

        iteration = iteration + 1;

        dX = next(mbq);

        % Output
        dY = dX(size(X,2)+1,:);
        % Pesi (inutili per ora)
        dW = dX(size(X,2)+2,:);
        % Inputs
        dX = dX(1:size(X,2),:);

        % Richiamo il model gradient, calcolo la loss e i gradienti        
        [gradients,Loss,Losses] = dlfeval(accfun,parameters,dX,dY,RF,alpha,dW);

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
    LossTrain = double(extractdata(gather(mean((dY-dYp).^2))));
    LossTest = mean((Ytest(ktest,:)-Yptest).^2);

    % questi sono i pesi del primo layer per fare feature selection
    W = double(extractdata(gather(parameters.W.weights)));

    %%
    
    % in questa sezione calcolo come sta evolvendo (statisticamente) la
    % loss nel training e test set e lo utilizzo per aggiornare alpha (ne
    % parliamo poi, per ora non perdeteci tempo)

    LastLTrains(1:end-1) = LastLTrains(2:end);
    LastLTests(1:end-1) = LastLTests(2:end);

    LastLTrains(end) = LossTrain;
    LastLTests(end) = LossTest;

    L_train_med = median(LastLTrains); 
    L_test_med = median(LastLTests); 

    L_train_std = std(LastLTrains); 
    L_test_std = std(LastLTests); 

    Z_losses = (L_test_med - L_train_med)./(L_test_std.^2 + L_train_std.^2).^0.5;

    if epoch >= 2000

        alpha = alpha.*(1 + tanh(Z_losses-Z_threshold).*dalpha);

    end

    %% Faccio vari plot

    figure(3)
    subplot(2,3,1)
    hold on
    plot(epoch,Loss,'.b','markersize',16)
    plot(epoch,Losses(1),'.r','markersize',12)
    plot(epoch,Losses(2),'.k','markersize',12)
    set(gca,"YScale",'log')
    grid on
    grid minor
    xlabel("epoch")
    ylabel("losses")
    legend("Loss","Loss MSE","Loss VES")

    subplot(2,3,2)
    hold on
    errorbar(epoch,L_train_med,L_train_std,'.b','markersize',16)
    errorbar(epoch,L_test_med,L_test_std,'.r','markersize',16)
    set(gca,"YScale",'log')
    grid on
    grid minor
    xlabel("epoch")
    ylabel("losses")
    legend("Train","Test")

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
    xticklabels(Xlabel)
    ylabel("VES")

    drawnow

end

%% Save
% cambiate pure nome file come vi fa più comodo
save("Results and Comparisons\Results_01.mat","X","Y","Xlabel","Ylabel",...
    "parameters","LossTest","LossTrain","alpha","averageSqGrad","averageGrad",...
    "Norm","L0","D0")

