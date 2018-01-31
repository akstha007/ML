%Assignement 4.e: Backpropagation Algorithm with 2 hidden layers

clear;
clc;

main();

%Squashing function
function sq_val = squashing(z)
    sq_val = 1./(1+exp(-z));
end


function main()
    %Initial variables
    x = [0 0; 0 1; 1 0; 1 1]; %Input
    t = [0, 1, 1, 0];   %Ouput XOR
    n = size(x);
    m = length(t);
    alpha = 1;  %Learning rate
    iterations = 1;   %no. of iterations
    
    x0 = 0.25;  %Input Bias
    h0 = 0.25;  %Hidden layer 1 Bias
    z0 = 0.25;  %Hidden layer 2 Bias
    
    r0 = -0.5;  %random variable lower limit
    r1 = 0.5;   %random variable upper limit
    
    v = (r1 - r0) * rand(2,3) + r0;
    h = (r1 - r0) * rand(3,2) + r0;
    w = (r1 - r0) * rand(2,1) + r0;    
    
    %Iterating the operations
    
    for i = 1:iterations
        
        z_in = (x * v) + x0   
        z = squashing(z_in)
                
        h_in = (z * h) + h0
        hh = squashing(h_in)
        
        y_in = ((hh * w) + z0)'
        y = squashing(y_in)
        error = t - y
        
        f_dash = y' * (1 - y)
        delta = error * f_dash    %weight correction term
        delta_weight = alpha * delta * hh
        delta_weight_bias = alpha * delta
        w = w + delta_weight'
        z0 = z0 + delta_weight_bias
        err_graph(i) = sum(error);
        
        f_dash1 = hh * (1 - hh)'
        delta1 = sum(delta * h)   %weight correction term hidden layer
        delta_weight1 = alpha * delta1 * z
        delta_weight_bias1 = alpha * delta1
        h = h + delta_weight1'
        h0 = h0 + delta_weight_bias1
        
        f_dash_output = z_in .* (1 - z_in)
        delta_output = sum(delta .* h) .* f_dash_output
        delta_v = alpha * delta_output * x
        delta_v_bias = alpha * delta_output
        v = v + delta_v'
        x0 = x0 + delta_v_bias
    end
    
    plot(0:iterations-1, err_graph(1:iterations));
    title('Sigmoid');
    ylabel('Error');
    xlabel('No. of iterations');
    
end