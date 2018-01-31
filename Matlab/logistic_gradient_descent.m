function [theta, j_list] = logistic_gradient_descent(x,y,m,theta,alpha,iterations)
    %preparing j_list which contains list of j_theta cost function values
    j_list = zeros(iterations,1);
    
    for i = 1:iterations
        %hypothesis
        h = logistic_hypothesis(x,theta);
        
        theta1 = theta(1) - alpha * sum((h - y) .* x(:,1)) / m;
        theta2 = theta(2) - alpha * sum((h - y) .* x(:,2)) / m;
        
        theta(1) = theta1;
        theta(2) = theta2;
        
        j_list(i) = logistic_cost(x,y,m,theta);
     end
end