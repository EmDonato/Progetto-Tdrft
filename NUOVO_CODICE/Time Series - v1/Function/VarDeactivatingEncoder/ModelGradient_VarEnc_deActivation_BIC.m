function [gradients,Loss,MSE,K] = ModelGradient_VarEnc_deActivation_BIC(parameters,dX,dY,RF,Weights)


    [dYp,parameters] = VarEnc_deAct(dX,dY(:,1),1,parameters,RF);

    %% Loss prediction

    MSE = sum(Weights.*(dY-dYp).^2)./sum(Weights);
    
    %% Loss deActivation

    W = parameters.W.weights;

    K = sum(abs(W/RF),'all');

    %% Losses 

    N = size(dYp,2);
     
    Loss = N.*log(MSE) + K.*log(N);
    
    gradients = dlgradient(Loss,parameters);

end