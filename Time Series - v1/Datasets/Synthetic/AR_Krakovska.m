%% AR Krakovska

clear; clc;

%%

N = 120000;

C = 0.6;

sigmax = 0.01;
sigmay = 0.01;

x = normrnd(0,sigmax,1,N);
y = normrnd(0,sigmay,1,N);

for i = 2 : N

    x(i) = 0.5*x(i-1) + 0.2*y(i-1) + normrnd(0,sigmax);
    y(i) = C*x(i-1) + 0.7*y(i-1) + normrnd(0,sigmay);

end

X(1,:) = x(1,2e4+1:N);
X(2,:) = y(1,2e4+1:N);

save("AR_Krakovska_C_=_"+C+".mat","X","C")
