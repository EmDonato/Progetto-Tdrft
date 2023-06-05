%% Pearson Regression based methodology

function Mr = FS_PearsonCorrelation(X,Y)

    rx = abs(corr(X));
    rft = abs(corr(X,Y));
    
    I = eye(size(rx));
    
    rx = rx - I;
    
    k = size(rx,1);
    
    rff = sum(rx,2);
    
    Mr = k.*rft./(k+k*(k-1)*rff);

end


