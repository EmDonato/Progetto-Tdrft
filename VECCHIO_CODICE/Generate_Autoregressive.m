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

%% initialisation

X = randn(M,N)*0.1;

Xlabel = "x"+(1:M)';

t = 1:N;

%% simulation

for i = 2 : N

    X(:,i) = sum(I.*X(:,i-1),1)' + randn(M,1)*0.1; % sum(X,1) somma le colonne del vettore X

end

%% plot

figure(1)
clf
for i = 1 : M
    subplot(floor(M/2),floor(M/floor(M/2))+1,i)
    plot(X(i,:),'b')
end



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

%% initialisation

X = randn(M,N)*0.1;

Xlabel = "x"+(1:M)';

t = 1:N;

%% simulation

for i = 2 : N

    X(:,i) = sum(I.*X(:,i-1),1)' + randn(M,1)*0.1; % sum(X,1) somma le colonne del vettore X

end

%% plot

figure(1)
clf
for i = 1 : M
    subplot(floor(M/2),floor(M/floor(M/2))+1,i)
    plot(X(i,:),'b')
end



save('Autoregressive_v2.mat');

%% configuration 3
clear all; clc;

N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

%% initialisation

X = randn(M,N)*0.1;

Xlabel = "x"+(1:M)';

t = 1:N;

%% simulation

for i = 2 : N

    X(:,i) = sum(I.*X(:,i-1),1)' + randn(M,1)*0.1; % sum(X,1) somma le colonne del vettore X

end

%% plot

figure(1)
clf
for i = 1 : M
    subplot(floor(M/2),floor(M/floor(M/2))+1,i)
    plot(X(i,:),'b')
end



save('Autoregressive_v3.mat');


%% configuration 4
clear all; clc;
N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(2,1) = 0.1;
I(3,2) = 1;
%% initialisation

X = randn(M,N)*0.1;

Xlabel = "x"+(1:M)';

t = 1:N;

%% simulation

for i = 2 : N

    X(:,i) = sum(I.*X(:,i-1),1)' + randn(M,1)*0.1; % sum(X,1) somma le colonne del vettore X

end

%% plot

figure(1)
clf
for i = 1 : M
    subplot(floor(M/2),floor(M/floor(M/2))+1,i)
    plot(X(i,:),'b')
end



save('Autoregressive_v4.mat');

%% configuration 5
clear all; clc;
N = 1e4;
M = 3;

% I = interaction matrix
I = zeros(M,M);

% I(i,j) significa che i causa j con intensità I(i,j) 

I(2,1) = 0.2;
I(3,1) = 1;

%% initialisation

X = randn(M,N)*0.1;

Xlabel = "x"+(1:M)';

t = 1:N;

%% simulation

for i = 2 : N

    X(:,i) = sum(I.*X(:,i-1),1)' + randn(M,1)*0.1; % sum(X,1) somma le colonne del vettore X

end

%% plot

figure(1)
clf
for i = 1 : M
    subplot(floor(M/2),floor(M/floor(M/2))+1,i)
    plot(X(i,:),'b')
end



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

%% initialisation

X = randn(M,N)*0.1;

Xlabel = "x"+(1:M)';

t = 1:N;

%% simulation

for i = 2 : N

    X(:,i) = sum(I.*X(:,i-1),1)' + randn(M,1)*0.1; % sum(X,1) somma le colonne del vettore X

end

%% plot

figure(1)
clf
for i = 1 : M
    subplot(floor(M/2),floor(M/floor(M/2))+1,i)
    plot(X(i,:),'b')
end



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

%% initialisation

X = randn(M,N)*0.1;

Xlabel = "x"+(1:M)';

t = 1:N;

%% simulation

for i = 2 : N

    X(:,i) = sum(I.*X(:,i-1),1)' + randn(M,1)*0.1; % sum(X,1) somma le colonne del vettore X

end

%% plot

figure(1)
clf
for i = 1 : M
    subplot(floor(M/2),floor(M/floor(M/2))+1,i)
    plot(X(i,:),'b')
end



save('Autoregressive_v7.mat');


%% configuration 8
clear all; clc;


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
%% initialisation

X = randn(M,N)*0.1;

Xlabel = "x"+(1:M)';

t = 1:N;

%% simulation

for i = 2 : N

    X(:,i) = sum(I.*X(:,i-1),1)' + randn(M,1)*0.1; % sum(X,1) somma le colonne del vettore X

end

%% plot

figure(1)
clf
for i = 1 : M
    subplot(floor(M/2),floor(M/floor(M/2))+1,i)
    plot(X(i,:),'b')
end



save('Autoregressive_v8.mat');

