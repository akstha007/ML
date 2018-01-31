clear;
clc;

%read the trainig data
data = load('training_data.txt');

%initializing the variables
x = data(:,1);   %features
y = data(:,2);   %actual result
alpha = 0.1;   %learning rate
m = length(y);   %no. of training data
iterations = 1500;   %no. of iterations for computing gradient descent
theta = zeros(2,1);  %intial parameters

%plot the data
plot(x,y,'b.','MarkerSize',6)
title('Assignment 2')
xlabel('Hours Studied')
ylabel('Result (1-pass, 0-fail)')

%add column ones to X for x0
x = [ones(m,1), x];

%print alpha, intial theta and iterations
fprintf('Alpha: %d\n',alpha)
fprintf('Iterations: %d\n', iterations)
fprintf('Initial Theta-1: %d\n', theta(1))
fprintf('Initial Theta-2: %d\n\n', theta(2))

%compute linear gradient descent
[linear_theta, linear_j_list] = linear_gradient_descent(x,y,m,theta,alpha,iterations);
fprintf('Linear Regression:\n')
fprintf('Theta-1: %d\nTheta-2: %d\n',linear_theta(1), linear_theta(2))
fprintf('Hypothesis: y = %d + %d * x1\n',linear_theta(1), linear_theta(2))

%Compute Linear regression Accuracy
linear_accuracy = linear_prediction(x,y,linear_theta);
fprintf('Accuracy: %d %%\n',linear_accuracy)

%compute logistic gradient descent
[logistic_theta, logistic_j_list] = logistic_gradient_descent(x,y,m,theta,alpha,iterations);
fprintf('\nLogistic Regression:\n')
fprintf('Theta-1: %d\nTheta-2: %d\n',logistic_theta(1), logistic_theta(2))
fprintf('Hypothesis: y = 1/(1 + exp(%d + %d * x1))\n',logistic_theta(1), logistic_theta(2))

%Compute Logistic regression accuracy
logistic_accuracy = logistic_prediction(x,y,logistic_theta);
fprintf('Accuracy: %d %%\n',logistic_accuracy)

%plotting linear regression line
hold on;
plot(x(:,2),x*linear_theta,'-')

%plotting logistic regression line
plot(x(:,2),logistic_hypothesis(x,logistic_theta),'-');
legend('Training data', 'Linear Regression', 'Logistic Regression');
%legend('Training data','Logistic Regression');
hold off;

%plot cost function vs iterations
%plot(1:iterations, linear_j_list, '-')
%plot(1:iterations, logistic_j_list, '-')
%legend('Linear Cost', 'Logistic Cost')
%title('Linear Regression')
%xlabel('Iterations')
%ylabel('Cost function')

