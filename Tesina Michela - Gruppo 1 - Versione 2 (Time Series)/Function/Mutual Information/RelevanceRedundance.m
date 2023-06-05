
function [REL_Y,MI_X,MI_Y] = RelevanceRedundance(X,Y)

bin = floor(size(X,1).^0.5/2^0.5);

for i = 1 : size(X,2)
    for j = 1 : size(X,2)

        k = randsample(size(X,1),size(X,1));

        MI_X_ref = Mutual_information_Shannon(X(:,i),X(k,i),bin,bin);

        MI_X(i,j) = Mutual_information_Shannon(X(:,i),X(:,j),bin,bin) - MI_X_ref;

    end
end

k = randsample(size(X,1),size(X,1));

MI_Y_ref = Mutual_information_Shannon(Y(:,1),Y(k,1),bin,bin);

for i = 1 : size(X,2)


    MI_Y(1,i) = Mutual_information_Shannon(Y,X(:,i),bin,bin)-MI_Y_ref;

    MI_X_temp = MI_X(:,i);
    MI_X_temp(i) = 0;

    REL_Y(1,i) = MI_Y(1,i) - sum(MI_X_temp);

end

end



