%% generate Autoregressive

clear; clc;

ParameterC1 = 0.01;
ParameterC2 = 0.02;
ParameterC3 = 0.1;
ParameterC4 = 0.3;
ParameterC5 = 0.5;   %ParameterC5bis = 0.532; %Sbaglia ma funziona
ParameterC6 = 0.8;   %Non funziona,Diverge
ParameterC7 = 1;     %Non funziona,Diverge


%% configuration ParameterC1

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 0.2;
I(1,2) = 0.5;
I(1,3) = 0;

I(2,1) = 0;
I(2,2) = ParameterC1;
I(2,3) = 1;

I(3,1) = 1;
I(3,2) = 0;
I(3,3) = 0.2;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_vc1.mat');

%% configuration ParameterC2

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 0.2;
I(1,2) = 0.5;
I(1,3) = 0;

I(2,1) = 0;
I(2,2) = ParameterC2;
I(2,3) = 0.3;

I(3,1) = 2;
I(3,2) = 0;
I(3,3) = 0.2;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_vc2.mat');

%% configuration ParameterC3

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 0.2;
I(1,2) = 0.5;
I(1,3) = 0;

I(2,1) = 0;
I(2,2) = ParameterC3;
I(2,3) = 0.3;

I(3,1) = 2;
I(3,2) = 0;
I(3,3) = 0.2;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_vc3.mat');

%% configuration ParameterC4

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 0.2;
I(1,2) = 0.5;
I(1,3) = 0;

I(2,1) = 0;
I(2,2) = ParameterC4;
I(2,3) = 0.3;

I(3,1) = 2;
I(3,2) = 0;
I(3,3) = 0.2;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_vc4.mat');

%% configuration ParameterC5

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 0.2;
I(1,2) = 0.5;
I(1,3) = 0;

I(2,1) = 0;
I(2,2) = ParameterC5;
I(2,3) = 0.3;

I(3,1) = 2;
I(3,2) = 0;
I(3,3) = 0.2;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_vc5.mat');

%% configuration ParameterC6

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 0.2;
I(1,2) = 0.5;
I(1,3) = 0;

I(2,1) = 0;
I(2,2) = ParameterC6;
I(2,3) = 0.3;

I(3,1) = 2;
I(3,2) = 0;
I(3,3) = 0.2;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_vc6.mat');

%% configuration ParameterC7

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 0.2;
I(1,2) = 0.5;
I(1,3) = 0;

I(2,1) = 0;
I(2,2) = ParameterC7;
I(2,3) = 0.3;

I(3,1) = 2;
I(3,2) = 0;
I(3,3) = 0.2;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_vc7.mat');