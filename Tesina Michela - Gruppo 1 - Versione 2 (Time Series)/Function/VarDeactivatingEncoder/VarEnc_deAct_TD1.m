%% Net 01

function [Yp,parameters] = VarEnc_deAct_TD1(X,Y,Predict,parameters,RF,delays)


    %% Net configuration (only fullyconnet)

    Layer = [20 20 20 20 20 size(Y,1)];

    delays = length(delays);

if Predict == 0

    parameters = [];

    %% TDNN Weight Layer

    M = (size(X,1)/delays);

    parameters.W.weights = dlarray(ones(M,1));

    for i = 1 : M

        X((i-1)*delays+1:i*delays,:) = ...
            (parameters.W.weights(i) + randn(size(X((i-1)*delays+1:i*delays,:)))*RF)...
            .*X((i-1)*delays+1:i*delays,:);

    end

    %% Encoder

    for i = 1 : length(Layer) - 1

        parameters.("l"+i).weights = dlarray(randn([Layer(i) length(X)])/10);
        parameters.("l"+i).bias = dlarray(randn([Layer(i) 1])/20);

        X = fullyconnect(X,parameters.("l"+i).weights,parameters.("l"+i).bias);

        X = tanh(X);

    end

    i = i + 1;

    parameters.("Output").weights = dlarray(randn([Layer(i) length(X)]));
    parameters.("Output").bias = dlarray(randn([Layer(i) 1]));

    Yp = fullyconnect(X,parameters.("Output").weights,parameters.("Output").bias);


else


    %% TDNN Weight Layer

    M = (size(X,1)/delays);

    for i = 1 : M

        X((i-1)*delays+1:i*delays,:) = ...
            (parameters.W.weights(i) + randn(size(X((i-1)*delays+1:i*delays,:)))*RF)...
            .*X((i-1)*delays+1:i*delays,:);

    end

    %% Encoder

    for i = 1 : length(Layer) - 1

        X = fullyconnect(X,parameters.("l"+i).weights,parameters.("l"+i).bias);

        X = tanh(X);

    end

    Yp = fullyconnect(X,parameters.("Output").weights,parameters.("Output").bias);


end








