%% generate Autoregressive

clear all; clc;

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
clear all; clc;


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
clear all; clc;

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v3.mat');


%% configuration 4
clear all; clc;
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
clear all; clc;
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
clear all; clc;
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
clear all; clc;
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

%% configuration 8
clear all; clc;

N = 1e4;
M = 3;

c = 0;
% I = interaction matrix
I = zeros(M,M);
aus = 10;
% I(i,j) significa che i causa j con intensità I(i,j) 
for c = 0:0.1:0.8
    aus = aus*c;
    I(1,3) = 0.8;
    I(2,1) = 0.6;
    I(2,3) = 1.2;
    I(3,3) = 0.2;
    I(1,2) = c;
    X = calculate_autoregressive(M,N,I);
    save("Autoregressive_v8_C="+aus+".mat");
    aus = 10;
end
%% configuration 8
clear all; clc;
N = 1e4;
M = 3;

c = 0;
% I = interaction matrix
I = zeros(M,M);
aus = 10;
% I(i,j) significa che i causa j con intensità I(i,j) 
for c = 0:0.01:0.03
    aus = aus*c;
    I(1,3) = 0.8;
    I(2,1) = 0.6;
    I(2,3) = 1.2;
    I(3,3) = 0.2;
    I(1,2) = c;
    X = calculate_autoregressive(M,N,I);
    save("Autoregressive_v8_C="+aus+".mat");
    aus = 10;
end
%% configuration 9

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,3) = 0.2;
I(1,1) = 0.2;
I(1,2) = 0.2;

I(2,3) = 0.2;
I(2,1) = 0.2;
I(2,2) = 0.2;

I(3,3) = 0.2;
I(3,1) = 0.2;
I(3,2) = 0.2;
X = calculate_autoregressive(M,N,I);

save('Autoregressive_v9.mat');
%% configuration 2bis

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(1,3) = 1;
I(1,1) = 0.6;
I(1,2) = 0.6;

I(2,3) = 0;
I(2,1) = 0.2;
I(2,2) = 0;

I(3,3) = 0;
I(3,1) = 0;
I(3,2) = 0;

X = calculate_autoregressive(M,N,I);

save('Autoregressive_v2bis.mat');
