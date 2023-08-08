clear; clc;

%% Rössler → Lorenz


num_points = 21000;  % Numero di punti temporali desiderati
tspan = linspace(0, 30, num_points);  % Genera un vettore di punti temporali

x0 = [0.0; 0.0; 0.4; 0.3; 0.3; 0.3];            % Condizioni iniziali (vettore R3)

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

function dxdt = mySystem(t, x)
    C = 3;
    dxdt = zeros(6, 1);
    % Definisci le equazioni differenziali
    dxdt(1) = -6 * (x(2) + x(3));
    dxdt(2) = 6 * (x(1) + 0.2 * x(2));
    dxdt(3) = 6 * (0.2 + x(3) * (x(1) - 5.7));
    dxdt(4) = 10 * (-x(4) + x(5));
    dxdt(5) = 28 * x(4) - x(5) - x(4) * x(6) + C * x(2) * x(2);
    dxdt(6) = x(4) * x(5) - (8 / 3) * x(6);
end
