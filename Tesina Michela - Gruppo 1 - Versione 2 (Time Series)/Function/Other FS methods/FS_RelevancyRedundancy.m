%% FS mutual information

function [Score,SelectedFeatures] = FS_RelevancyRedundancy(X,Y)

bin = (length(Y)/2).^0.5;
STOP = 0;

%% Calculation of relevancy

candidates = 1:size(X,2);

for i = 1 : size(X,2)

    Relevancy(i,1) =  Mutual_information_Differential(X(:,i),Y,bin,bin);

end

[J_max,i_max] = max(Relevancy);

S = i_max;
JJ = J_max;

candidates(i_max) = [];

while STOP == 0

    clear Redundancy

    for i = 1 : length(candidates)

        for j = 1 : length(S)

            Redundancy(i,j) = Mutual_information_Differential(X(:,candidates(i)),X(:,S(j)),bin,bin);

        end

    end

    J = Relevancy(candidates) - sum(Redundancy,2)/length(S);

    [J_max,i_max] = max(J);



    S = [S candidates(i_max)];
    JJ = [JJ J_max];

    candidates(i_max) = [];

    if length(candidates) == 1

        S = [S candidates];
        JJ = [JJ min(J)];
        STOP = 1;

    end

end


SelectedFeatures = S;
Score = JJ;


end
