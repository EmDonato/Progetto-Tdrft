function [MI, H_xy] = Mutual_information_Shannon(x,y,binx,biny)

    x_min = min(x);
    x_max = max(x);
    dx = (x_max-x_min)/binx; 
    x_bin = x_min:dx:x_max;

    y_min = min(y);
    y_max = max(y);
    dy = (y_max-y_min)/biny; 
    y_bin = y_min:dy:y_max;

    c_x = histcounts(x,x_bin);
    p_x = c_x/(sum(c_x));
    p_x(p_x==0) = eps;

    c_y = histcounts(y,y_bin);
    p_y = c_y/(sum(c_y));
    p_y(p_y==0) = eps;

    c_xy = histcounts2(x,y,x_bin,y_bin);
    p_xy = c_xy/(sum(sum(c_xy)));
    p_xy(p_xy==0)=eps;

    H_xy = -sum(sum(p_xy.*log(p_xy)));

    MI = sum(sum(p_xy.*log(p_xy./(p_x'.*p_y))));

end


