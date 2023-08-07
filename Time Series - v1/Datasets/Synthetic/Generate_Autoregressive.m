%% generate Autoregressive

clear; clc;

%% configuration 1

N = 1e4;
M = 6;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 0.95;
I(2,2) = 0;
I(3,3) = 0.99;
I(4,4) = 0.98;
I(5,5) = 0;
I(6,6) = 0.81;

I(2,1) = -0.4;
I(3,2) = 0.4;

I(5,6) = 0.4;
I(6,5) = 0.4;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v1.mat');

%% configuration 2

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,1) = 1;
I(2,2) = 1;
I(3,3) = 1;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v2.mat');

%% configuration 3

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v3.mat');


%% configuration 4

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(2,1) = 1;
I(3,2) = 1;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v4.mat');

%% configuration 5

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(2,1) = 1;
I(3,1) = 1;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v5.mat');

%% configuration 6

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,2) = 1;
I(2,1) = 1;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v6.mat');

%% configuration 7

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,3) = 1;
I(2,1) = 1;
I(3,2) = 1;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v7.mat');

