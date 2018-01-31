function j_theta = linear_cost(x,y,m,theta)
    %hypothesis
    h = x*theta;

    %calculate the cost function
    j_theta = (1./(2*m)) * sum((h - y).^2);

end