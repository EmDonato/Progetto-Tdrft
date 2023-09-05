%% Net 01

function [Yp,parameters] = VarEnc_deAct(X,Y,Predict,parameters,RF)


    %% Net configuration (only fullyconnet)

    Layer = [20 20 20 size(Y,1)];

if Predict == 0

    parameters = [];

    %% Weight Layer

    parameters.W.weights = dlarray(ones(size(X(:,1))));

    X = X.*(parameters.W.weights + randn(size(X))*RF);


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

    %% Weight Layer

    X = X.*(parameters.W.weights + randn(size(X))*RF);

    %% Encoder

    for i = 1 : length(Layer) - 1

        X = fullyconnect(X,parameters.("l"+i).weights,parameters.("l"+i).bias);

        X = tanh(X);

    end

    Yp = fullyconnect(X,parameters.("Output").weights,parameters.("Output").bias);


end








