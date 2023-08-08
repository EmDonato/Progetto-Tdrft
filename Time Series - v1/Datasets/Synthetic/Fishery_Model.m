%% Fishery_Model

clear; clc;

%%

N = 120000;

C = 0.6;

sigmax = 0.01;
sigmay = 0.01;

Rx = zeros(1,N);
x = zeros(1,N);
y = zeros(1,N);
Ry = zeros(1,N);
for i = 1 : 4 
    Rx(i) = 0.5;
    Ry(i) = 0.5;
    x(i) = 0.5;
    y(i) = 0.5;
end
Z = redNoise(1,N);
plot(Z);
for i = 5 : N
    Rx(i) = x(i-1)*(3.1*(1 - x(i-1)))*exp(0.5*Z(i));
    Ry(i) = y(i-1)*(2.9 *(1 - y(i-1)))*exp(0.6*Z(i));
    x(i) = 0.4*x(i-1) + max(Rx(i - 4),0);
    y(i) = 0.35*y(i-1) + max(Ry(i- 4),0);

end

X(3,:) = x(1,1000+1:N);
X(4,:) = y(1,1000+1:N);
X(1,:) = Rx(1,1000+1:N);
X(2,:) = Ry(1,1000+1:N);

save("Fishery_Model=_"+C+".mat","X","C")
