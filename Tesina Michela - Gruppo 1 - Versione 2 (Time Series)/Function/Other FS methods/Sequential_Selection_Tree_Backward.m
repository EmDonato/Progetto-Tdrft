
function [SelectedFeatures_Regression,history] = Sequential_Selection_Tree_Backward(X,Y)

    ind = 1:size(Y,1);
    
    myfun = @(XTrain,yTrain,XTest,yTest) ...
      size(XTest,1)*loss(fitrtree(XTrain,yTrain),XTest,yTest);
    
    cv = cvpartition(Y(ind),"KFold",10);
    
    opts = statset("Display","iter");
    [SelectedFeatures_Regression,history] = sequentialfs(myfun,X(ind,:),Y(ind),"CV",cv,"Direction","backward","Options",opts);

end

