%Assignement 4 a vs d: Backpropagation Algorithm

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
    iterations = 100;   %no. of iterations
    mue = 0.5;  %momentum term
    sigma = 1;
    
    x0 = 0.25;  %Input Bias
    z0 = 0.25;  %Hidden layer Bias
    
    r0 = -0.5;  %random variable lower limit
    r1 = 0.5;   %random variable upper limit
    
    v = (r1 - r0) * rand(2,4) + r0;
    w = (r1 - r0) * rand(4,1) + r0;
    
    old_w = w;
    old_v = v;
    old_x0 = x0;
    old_z0 = z0;
    
    %Iterating the operations
    
    for i = 1:iterations
        
        z_in = sum(x * v) + x0;        
        z = squashing(z_in);
                
        y_in = sum(z .* w) + z0;
        y = squashing(y_in);
        error = t - y;
        
        f_dash = y' * (1 - y);
        delta = error * f_dash;    %weight correction term
        delta_weight = alpha * delta' * z;
        delta_weight_bias = alpha * delta;
        w = w + delta_weight;
        z0 = z0 + delta_weight_bias;
        err_graph(i) = sum(error);
        
        f_dash_output = z_in .* (1 - z_in);
        delta_output = sum(delta .* w) .* f_dash_output;
        delta_v = alpha * delta_output * x;
        delta_v_bias = alpha * delta_output;
        v = v + delta_v';
        x0 = x0 + delta_v_bias;
    end
    
    hold on;
    %plot(0:iterations-1, err_graph(1:iterations),'r.');
    title('Sigmoid');
    ylabel('Error');
    xlabel('No. of iterations');    
    
    w = old_w;
    v = old_v;
    x0 = old_x0;
    z0 = old_z0;
    %Iterating the operations
    
    n = 2;
    h = 4;  %no. of hidden units
    
    beta = 0.7 * power(h,1/n);
    
    r0 = -0.5;  %random variable lower limit
    r1 = 0.5;   %random variable upper limit
    
    v = (r1 - r0) * rand(2,4) + r0;
    w = (r1 - r0) * rand(4,1) + r0;
    
    v = beta * v / norm(v);
    x0 = (beta+beta) * rand() - beta;
    
    %Iterating the operations
    
    for i = 1:iterations
        
        z_in = sum(x * v) + x0;        
        z = squashing(z_in);
                
        y_in = sum(z .* w) + z0;
        y = squashing(y_in);
        error = t - y;
        
        f_dash = y' * (1 - y);
        delta = error * f_dash;    %weight correction term
        delta_weight = alpha * delta' * z;
        delta_weight_bias = alpha * delta;
        w = w + delta_weight;
        z0 = z0 + delta_weight_bias;
        err_graph1(i) = sum(error);
        
        f_dash_output = z * (1 - z)';
        delta_output = sum(delta * w) * f_dash_output;
        delta_v = alpha * delta_output * x;
        delta_v_bias = alpha * delta_output;
        v = v + delta_v';
        x0 = x0 + delta_v_bias;
    end    
    
    plot(0:iterations-1, err_graph(1:iterations), 0:iterations-1, err_graph1(1:iterations));
    legend('a','b')
    hold off;
    
end