%% two_species

clear; clc;

%%
C = 0.8; 
N = 21000;

x = zeros(1,N);
y = zeros(1,N);

x(1) = 0.2;
y(1) = 0.4;

for t = 2 : N
    
    x(t) = x(t-1)*(3.78 - 3.78*x(t-1) - 0.07*y(t-1));
    y(t) = y(t-1)*(3.77 - 3.77*y(t-1) - 0.08*x(t-1));

end

X(1,:) = x(1,1000+1:N);
X(2,:) = y(1,1000+1:N);

save("two_species=_"+C+".mat","X","C")
