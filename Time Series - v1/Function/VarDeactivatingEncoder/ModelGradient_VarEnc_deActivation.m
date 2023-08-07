function [gradients,Loss,Losses] = ModelGradient_VarEnc_deActivation(parameters,dX,dY,RF,alpha0,Weights)


    [dYp,parameters] = VarEnc_deAct(dX,dY(:,1),1,parameters,RF);

    %% Loss prediction

    LossRec = sum(Weights.*(dY-dYp).^2)./sum(Weights);
    
    %% Loss deActivation

    W = parameters.W.weights;

    LossWeights = mean((W/RF).^2,'all');

%     LossWeights = mean(abs(W),'all');

    %% Losses 
     
    Losses = [LossRec LossWeights];
    
    alpha = [1 alpha0.*RF.^2];
    
    Losses = alpha.*Losses/sum(alpha);
    
    Loss = sum(Losses);
    
    gradients = dlgradient(Loss,parameters);

end