function dxdt = mySystem(t, x, y)
    % Definisci le equazioni differenziali per x
    dxdt = zeros(3, 1);
    
    dxdt(1) = -6 * (x(2) + x(3));
    dxdt(2) = 6 * (x(1) + 0.2 * x(2));
    dxdt(3) = 6 * (0.2 + x(3) * (x(1) - 5.7));
end

function dydt = mySystemY(t, x, y, C)
    % Definisci le equazioni differenziali per y
    dydt = zeros(3, 1);
    
    dydt(1) = 10 * (-y(1) + y(2));
    dydt(2) = 28 * y(1) - y(2) - y(1) * y(3) + C * x(2) * x(2);
    dydt(3) = y(1) * y(2) - (8 / 3) * y(3);
end
