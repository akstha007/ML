function accuracy = linear_prediction(x,y,theta)
    %prediction from linear regression
    h = round(x * theta);
    
    %checking predicted value with actual value for accuracy
    accuracy = mean(h == y)*100;
    
end