'''
Note:

Handle Error:

RuntimeWarning: overflow encountered in double_scalars
  j_theta = sum([(hypothesis(x[i],theta)-y[i])**2 for i in range(m)])/(2*m)

value overflow due to high alpha value

'''


import matplotlib.pyplot as plt
from numpy import ones
import numpy as np


x = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])
y = np.array([5.1,6.1,6.9,7.8,9.2,9.9,11.5,12.0,12.8])

#Hypothesis function
def hypothesis(x_row,theta):
    return (theta[0] + theta[1]*x_row)

#calculate cost funtion
def compute_cost(x,y,n,m,theta):
    j_theta = sum([(hypothesis(x[i],theta)-y[i])**2 for i in range(m)])/(2*m)
    return j_theta

#calculate new values of thetas using gradient descent
def gradient_descent(x,y,m,theta,alpha):
    n_theta = [0,0]
    n_theta[0] = theta[0] - alpha*(sum([(hypothesis(x[i],theta)-y[i]) for i in range(m)]))/(m)
    n_theta[1] = theta[1] - alpha*(sum([(hypothesis(x[i],theta)-y[i])*x[i] for i in range(m)]))/(m)
    return n_theta

#assume theta0 and theta1 values initially
theta = np.array([0.0,0.0])
iterations = 500
alpha = 1.0

j_list = []
t_list = []
x_arr = np.array([i for i in range(iterations)])


m = len(y)
n = len(theta)

print('Initial Parameters: \nTheta-0: {0:0.2f}, Theta-1: {1:0.2f}'.format(*theta))

for i in range(iterations):
    j_theta = compute_cost(x,y,n,m,theta)
    j_list.append(j_theta)
    theta = gradient_descent(x,y,m,theta,alpha)
    t_list.append(theta)    

print('Final Parameters: \nTheta-0: {0:0.2f}, Theta-1: {1:0.2f}'.format(*theta))

plt.title('Simple Linear Regression')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.scatter(x,y,label="Actual Value",color="g",marker="o",s=1)

approx_value = [hypothesis(x_row,theta) for x_row in x]
plt.plot(x, approx_value, label="Predicted Value", color='r', linewidth = 1)

plt.legend()
plt.show()

zipped = zip(*t_list)
#plt.plot(x_arr, zipped[0], label="Theta_0 Value", color='m', linewidth = 1)
#plt.plot(x_arr, zipped[1], label="Theta_1 Value", color='b', linewidth = 1)

plt.title('Gradient Descent')
plt.xlabel('Theta-0')
plt.ylabel('Theta-1')
plt.plot(zipped[0], zipped[1], label="Theta Values", color='g', linewidth = 1)
#plt.scatter(zipped[0], zipped[1],label="Theta Values",color="b",marker="o",s=1)
plt.legend()
plt.show()

plt.title('Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost function')
plt.plot(x_arr, j_list, label="J-Theta", color='m', linewidth = 1)
#plt.scatter(x_arr, j_list,label="J-Theta",color="m",marker="o",s=1)
plt.legend()
plt.show()

