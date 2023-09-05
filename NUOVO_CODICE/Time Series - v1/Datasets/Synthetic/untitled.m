clear; clc;

%%  Hénon → Hénon

N = 21000;

C = 0.7; %(0.7 to 0.8)

sigmax = 0.01;
sigmay = 0.01;

x1 = zeros(N);
x2 = zeros(N);
y1 = zeros(N);
y2 = zeros(N);

%inizializzazione
x1(1) = 0.7;
x2(1) = 0.0;
y1(1) = 0.91;
y2(1) = 0.7;

for t = 2 : N

x1(t) = 1.4 -x1(t-1)*x1(t-1) + 0.3*x2(t-1);
x2(t) = x1(t-1);
y1(t) = 1.4 - (C*x1(t-1)*y1(t-1) + (1-C)*y1(t-1)*y1(t-1)) + 0.3*y2(t-1);
y2(t) = y1(t-1);
end

X(1,:) = x(1,2e4+1:N);
X(2,:) = y(1,2e4+1:N);
X(3,:) = x(1,2e4+1:N);
X(4,:) = y(1,2e4+1:N);

save("Hénon_to_Hénon=_"+C+".mat","X","C")