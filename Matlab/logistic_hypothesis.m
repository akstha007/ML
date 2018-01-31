function h = logistic_hypothesis(x, theta)
    z = x * theta;
    h = 1./(1 + exp(-z));
end