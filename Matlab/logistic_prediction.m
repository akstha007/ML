function accuracy = logistic_prediction(x,y,theta)
    %prediction from linear regression
    h = round(logistic_hypothesis(x, theta));
    
    %checking predicted value with actual value for accuracy
    accuracy = mean(double(h == y))*100;
    
end