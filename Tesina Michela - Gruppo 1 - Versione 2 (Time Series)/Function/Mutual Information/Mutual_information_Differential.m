function [MI, H_xy] = Mutual_information_Differential(x,y,binx,biny)

    x_min = min(x);
    x_max = max(x);
    dx = (x_max-x_min)/binx; 
    x_bin = x_min:dx:x_max;

    y_min = min(y);
    y_max = max(y);
    dy = (y_max-y_min)/biny; 
    y_bin = y_min:dy:y_max;

    c_x = histcounts(x,x_bin);
    p_x = c_x/(sum(c_x)*dx);
    p_x(p_x==0) = eps;

    c_y = histcounts(y,y_bin);
    p_y = c_y/(sum(c_y)*dy);
    p_y(p_y==0) = eps;

    c_xy = histcounts2(x,y,x_bin,y_bin);
    p_xy = c_xy/(sum(sum(c_xy))*dx*dy);
    p_xy(p_xy==0)=eps;

    H_xy = -sum(sum(p_xy.*log(p_xy)))*dx*dy;

    MI = sum(sum(p_xy.*log(p_xy./(p_x'.*p_y))*dx*dy));

%     figure(4)
%     clf
%     subplot(1,2,1)
%     contourf(x_bin(1:end-1),y_bin(1:end-1),p_xy',100,'linestyle','none')
%     xlabel('x')
%     ylabel('y')
%     title('classic pdf')
%     colormap('jet')
    
end


