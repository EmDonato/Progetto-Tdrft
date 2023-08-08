clear; clc;

%% Rössler 0.5 → Rössler 2.5

C = 1;
num_points = 21000;  % Numero di punti temporali desiderati
tspan = linspace(0, 30, num_points);  % Genera un vettore di punti temporali

x0 = [0; 0; 0.4;
0; 0; 0.4];            % Condizioni iniziali (vettore R3)

% Risolvi il sistema di equazioni differenziali
[t, X] = ode45(@mySystem, tspan, x0);

% Estrai le variabili dalla soluzione
x1 = X(:, 1);
x2 = X(:, 2);
x3 = X(:, 3);
x4 = X(:, 4);
x5 = X(:, 5);
x6 = X(:, 6);


% Plot delle variabili rispetto al tempo
figure;
plot(t, x1, 'r', t, x2, 'g', t, x3, 'b',t, x4, 'y',t, x5, 'c',t, x6, 'm');
xlabel('Tempo');
ylabel('Valore delle variabili');
legend('x1', 'x2', 'x3', 'x4', 'x5', 'x6');
title('Variabili x1, x2, x3, x4, x5, x6 rispetto al tempo');

 X = X';
 X = X(:,1000+1:num_points);
 save("Rössler_0.5_to_Rössler_2.5=_"+C+".mat","X","C")


function dxdt = mySystem(t, x)
    C = 1;
    w1 = 0.5; w2 = 2.515;
    dxdt = zeros(6, 1);
    % Definisci le equazioni differenziali
    dxdt(1) = - w1*x(2) - x(3);
    dxdt(2) = w1*x(1) + 0.15*x(2);
    dxdt(3) = 0.2 + x(3)*(x(1) - 10);
    dxdt(4) = -w2*x(5) - x(6) + C*(x(1) - x(4));
    dxdt(5) = w2*x(4) + 0.72*x(5);
    dxdt(6) = 0.2 + x(6)*(x(4) - 10);
end
