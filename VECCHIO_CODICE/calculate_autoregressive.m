
function X = calculate_autoregressive(M,N,I)

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

end