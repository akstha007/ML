function j_theta = logistic_cost(x,y,m,theta)
    %hypothesis
    h = logistic_hypothesis(x,theta);
    
    %calculate the cost function
    j_theta = -1/m * sum(y .* log(h) + (1 - y) .* log(1 - h));
   

end