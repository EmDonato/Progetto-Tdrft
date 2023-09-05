function [gradients,Loss,Losses] = ModelGradient_VarEnc_deActivation(parameters,dX,dY,RF,alpha0,delays)


    [dYp,parameters] = VarEnc_deAct_TD1(dX,dY(:,1),1,parameters,RF,delays);

    %% Loss prediction

    LossRec = mean((dY-dYp).^2);
    
    %% Loss deActivation

    W = parameters.W.weights;

    LossWeights = mean(abs(W)/RF,'all').^2;

%     LossWeights = mean(abs(W),'all');

    %% Losses 
     
    Losses = [LossRec LossWeights];
    
    alpha = [1 alpha0.*RF.^2];
    
    Losses = alpha.*Losses/sum(alpha);
    
    Loss = sum(Losses);
    
    gradients = dlgradient(Loss,parameters);

end