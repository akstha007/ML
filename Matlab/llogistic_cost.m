function j_theta = llogistic_cost(x,y,m,theta)
    %hypothesis
    h = logistic_hypothesis(x,theta);
    
    %calculate the cost function
    %j_theta = 100./(m) * sum(y .* (h) + (1 - y) .* (1 - h));
    j_theta1 = (100./m) * sum((1 - y) .* (h));
    j_theta2 = (100./m) * sum(y .* (1 - h));
    j_theta = j_theta1 + j_theta2;
    %j_theta = (1./(2*m)) * sum((h - y).^2);

end