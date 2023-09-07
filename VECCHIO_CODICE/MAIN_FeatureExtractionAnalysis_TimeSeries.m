% gpu = gpuDevice();
% reset(gpu)
% 
% gpu = gpuDevice();
% disp(gpu)
% wait(gpu)
clear all; clc; warning off;

% parametri per i cicli
j_autoregressive_targhet = 1;
%NTarget = 6;
NTarget = 3;
W_old = zeros(NTarget);
figure(3);
W_matrix = zeros(NTarget,NTarget);
i_target = 1;
save('W_old.mat','W_old');
save('W_matrix.mat','W_matrix');
save('i_target.mat','i_target');


%%
% autoregressiveST = ['Autoregressive_v2.mat',"Autoregressive_v3.mat",'Autoregressive_v4.mat','Autoregressive_v5.mat','Autoregressive_v6.mat','Autoregressive_v7.mat','Autoregressive_v8.mat'];

for j_autoregressive_targhet = 1:7

    save('j_autoregressive_targhet.mat','j_autoregressive_targhet');
    i_target = 1;
    save('i_target.mat','i_target');

for i_target = 1 : NTarget
      


W_old = struct2array(load('W_old.mat'));
clear all; clc; warning off;
autoregressiveST = ['Autoregressive_v2.mat',"Autoregressive_v3.mat",'Autoregressive_v4.mat','Autoregressive_v5.mat','Autoregressive_v6.mat','Autoregressive_v7.mat','Autoregressive_v8.mat'];

    
% parametri per i cicli
Nepoch = 100;
%NTarget = 6;
NTarget = 3;
W_matrix = struct2array(load('W_matrix.mat'));
i_target = int32(struct2array(load('i_target.mat')));
j_autoregressive_targhet = int32(struct2array(load('j_autoregressive_targhet.mat')));
    disp('Valore di j_autoregressive_targhet: ');
    disp( j_autoregressive_targhet);
 fprintf('Il valore di i_target è %.2f\n', i_target);
figure(3)
clf

%% load data

load(autoregressiveST(j_autoregressive_targhet))

%% add paths

addpath("Function\VarDeactivatingEncoder\")

%% Prepare input and output

% Qua selezionate quale dei vostri input è da analizzare come output



% extract output
Y = X(i_target,:);
Ylabel = Xlabel(i_target);

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

%% Normalisation (normalizzo i dati)

Norm.Fmean = mean(Features,1);
Norm.Fstd = std(Features,[],1);

Norm.Tmean = mean(Targets,1);
Norm.Tstd = std(Targets,[],1);

Features = (Features' - Norm.Fmean')./Norm.Fstd';
Targets = (Targets' - Norm.Tmean')./Norm.Tstd';

%% dlarray (prepare data for training and prediction)

Train.minibatchsize = 1000;
Train.IterationPerEpoch = floor(size(Features,2)/Train.minibatchsize);

TrainingSet = 0.9;
NTrainingSet = floor(TrainingSet*length(Y));
ind = 1:NTrainingSet;

Train.ind_inputs = 1:size(Features,1);
Train.ind_outputs = (1:size(Targets,1)) + Train.ind_inputs(end);

ds = arrayDatastore([Features(:,ind)',Targets(:,ind)']);

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

RF = 0.1;

[Yp,parameters] = VarEnc_deAct_TD1(dX(Train.ind_inputs,1),dX(Train.ind_outputs,1),0,[],RF,delays);
[Yp,parameters] = VarEnc_deAct_TD1(dX(Train.ind_inputs,:),dX(Train.ind_outputs,1),1,parameters,RF,delays);

reset(mbq)

clear Yp

%% Training options

Z_threshold = 2;

iteration = 0;

averageGrad = [];
averageSqGrad = [];

L0 = 1e-2;
D0 = 1e-3;

alpha = 0.01;

%% Initialise train and test losses

LastLTrains = ones(10,1)*std(Y);
LastLTests = ones(10,1)*std(Y);

%% Model Gradient

accfun = dlaccelerate(@ModelGradient_VarEnc_deActivation);

%% Training
    for epoch = 1 : Nepoch
    
        reset(mbq)
        shuffle(mbq)
    
        RFt = 0.1;
    
        for i = 1 : Train.IterationPerEpoch
    
            iteration = iteration + 1;
    
            dX = next(mbq);
    
            dY = dX(Train.ind_outputs,:);
            dX = dX(Train.ind_inputs,:);
            
            [gradients,Loss,Losses] = dlfeval(accfun,parameters,dX,dY,RF,alpha,delays);
    
            % Update learning rate.
            learningRate = max(L0./(1 + D0*iteration),1e-4);
    
            % Update the network parameters using the adamupdate function.
            [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                averageSqGrad,iteration,learningRate);
    
        end
    
        %% Predict train and test
    
        [dYp,parameters] = VarEnc_deAct_TD1(dX,dY(:,1),1,parameters,0,delays);
    
        ktest = randsample(length(Ytest),min(length(Ytest),5e3));
        [dYptest,parameters] = VarEnc_deAct_TD1(dXtest(:,ktest),dY(:,1),1,parameters,0,delays);
    
        Yptest = double(extractdata(gather(dYptest)))';
    
        LossTrain = double(extractdata(gather(mean((dY-dYp).^2))));
        LossTest = mean((Ytest(ktest,:)-Yptest).^2);
    
        W = double(extractdata(gather(parameters.W.weights)));
    

        %%
        
        LastLTrains(1:end-1) = LastLTrains(2:end);
        LastLTests(1:end-1) = LastLTests(2:end);
    
        LastLTrains(end) = LossTrain;
        LastLTests(end) = LossTest;
    
        L_train_med = median(LastLTrains); 
        L_test_med = median(LastLTests); 
    
        L_train_std = std(LastLTrains); 
        L_test_std = std(LastLTests); 
    
        Z_losses = (L_test_med - L_train_med)./(L_test_std.^2 + L_train_std.^2).^0.5;
    
    
    
        %%
    
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
        ylabel("Losses")
        legend("Loss","Loss MSE","Loss VNE")
    
    
        subplot(2,3,2)
        hold on
        errorbar(epoch,L_train_med,L_train_std,'.b','markersize',16)
        errorbar(epoch,L_test_med,L_test_std,'.r','markersize',16)
        set(gca,"YScale",'log')
        grid on
        grid minor
        xlabel("epoch")
        ylabel("Losses")
        legend("Training","Test")
    
        subplot(2,3,3)
        hold off
        plot(dY,dYp,'.b','markersize',16)
        hold on
        plot(Ytest(ktest,:),Yptest,'.r','markersize',12)
        plot([-5 5],[-5 5],'-.k','LineWidth',2)
        grid on
        grid minor
        xlabel("Target")
        ylabel("Predicted")
        legend("Training","Test")
    
        subplot(2,3,[4 5 6])
        hold off
        plot(1:length(W),ones(size(W))*RF,'-.r','linewidth',2)
        hold on
        plot(1:length(W),ones(size(W))*RF*2,'--r','linewidth',2)
        plot(1:length(W),ones(size(W))*RFt,'-.k','linewidth',2)
        plot(1:length(W),abs(W),'ob','linewidth',2)
        ylim([0 inf])
        grid on
        grid minor
        xticks(1:length(W))
        xticklabels(Xlabel)
        ylabel("VES")
    
        drawnow
    
    W_old = struct2array(load('W_old.mat'));
    if(all(abs(abs(W_old)-abs(W))<0.001))
        fprintf('USCITOOOOOOOO');
        break
    end
    save('W_old.mat','W');

    W_display = abs(abs(W_old)-abs(W));
    disp(W_display);

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
%fprintf('Il valore di i_target è %.2f\n', i_target);
save('i_target.mat','i_target');
%W_matrix(W_matrix < 0.1) = 0;
%W_matrix(W_matrix > 0.1) = 1;
save('W_matrix.mat','W_matrix');

% Best.W = Best.W';
% save("Last_Results","Best")

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
    case 7
        save('I8.mat','I')
        save('W_matrix8.mat','W_matrix');
    otherwise
        disp("errore servono piu variabili");
end

save('j_autoregressive_targhet.mat','j_autoregressive_targhet');



end
end

%% solo se sono 8 le matrici


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
  W_matrix8 = struct2array(load('W_matrix8.mat'));
  ausW8colomn = W_matrix8(:);
  W_matrixTotal = horzcat(ausW2colomn,ausW3colomn,ausW4colomn,ausW5colomn,ausW6colomn,ausW7colomn,ausW8colomn);
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

  I8 = struct2array(load('I8.mat'));
  I8colomn = I8(:);

  ITotal = horzcat(I2colomn,I3colomn,I4colomn,I5colomn,I6colomn,I7colomn,I8colomn);
  ITotalColomn = ITotal(:);

%%




%inizializzazione parametri per il calcolo delle performance
%[TrueP,TrueN,FalseP,FalseN,W_graph] = PerformanceNew(W_matrixTotal,ITotal)

[soglia,AUC] = myROC(W_matrixTotal,ITotal);

%%
 CreateGraph(W_matrix7,I7,soglia)
% %%
% %inizializzazione parametri per il calcolo delle performance
% %[TrueP,TrueN,FalseP,FalseN,W_graph] = PerformanceNew(W_matrix,I)
% % Ottieni le coordinate degli archi nella matrice
% [righe, colonne] = find(W_graph);
% 
% % Crea il grafo orientato
% grafo = digraph(righe, colonne);
% 
% % Visualizza il grafo
% figure(99)
% plot(grafo);
% 


