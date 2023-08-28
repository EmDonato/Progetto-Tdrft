function [TrueP,TrueN,FalseP,FalseN] = Performance(w,threshold,I)
%CALCOLATE Summary of this function goes here
%   Detailed explanation goes here
TrueP = 0;
TrueN = 0;
FalseP = 0;
FalseN = 0;
d = size(w);
ausI = I;
for i = 1:d(1)
    if abs(ausI(i,1)) > 0
         ausI(i,1)  = 1;
    end
    if w(i,1) >= threshold
         w(i,1) = 1;
    else
         w(i,1) = 0;
    end
if ausI(i,1) - w(i,1) == 0 && ausI(i,1) == 1
   TrueP = TrueP + 1; 
elseif ausI(i,1) - w(i,1) == 1  
   FalseN = FalseN +1;
elseif ausI(i,1) - w(i,1) == -1
   FalseP = FalseP +1;
elseif ausI(i,1) - w(i,1) == 0 && ausI(i,1) == 0
   TrueN = TrueN + 1;
end 
end
end

