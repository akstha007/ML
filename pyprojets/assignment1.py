'''
------------------Assignment 1--------------------

Simple Linear Regression
========================

Submitted by: Ashok Kumar Shrestha
Date: Aug 29, 2017


'''

import matplotlib.pyplot as plt
import numpy as np

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

#Plot given data and predicted value in graph
def draw_xymodel(x,y,alpha):
    plt.title('Simple Linear Regression')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.scatter(x,y,label="Actual Value",color="g",marker="o",s=1)

    approx_value = [hypothesis(x_row,theta) for x_row in x]
    plt.plot(x, approx_value, label="Predicted Value for alpha="+str(alpha), color='r', linewidth = 1)

    plt.legend()
    plt.show()

#plot cost function in graph
def draw_cost_function(x_arr, j_list, xlable_name, alpha):
    plt.title('Cost Function for alpha='+str(alpha))
    plt.xlabel(xlable_name)
    plt.ylabel('Cost function')    
    plt.plot(x_arr, j_list, color='g', linewidth = 1)   
    plt.scatter(x_arr, j_list, color="m",marker="o",s=1) 
    #plt.legend()
    plt.show()

if __name__ == '__main__':

    x = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])
    y = np.array([5.1,6.1,6.9,7.8,9.2,9.9,11.5,12.0,12.8])

    np.seterr(all = 'raise')
    #assume theta0 and theta1 values initially
    theta = np.array([0.0,0.0])
    iterations = 1500
    alpha = 0.1
    
    j_list = []
    t_list = []
    x_arr = []
    m = len(y)
    n = len(theta)
    is_converged = True;

    print('Alpha: ' + str(alpha))
    print('-----------Initial Parameters-----------')
    print('Theta-0: {0:0.2f} \nTheta-1: {1:0.2f}'.format(*theta))

    try:
        for i in range(iterations):
            j_theta = compute_cost(x,y,n,m,theta)
            j_list.append(j_theta)
            theta = gradient_descent(x,y,m,theta,alpha)
            t_list.append(theta)    
    except FloatingPointError as e:
        #print(e)   
        is_converged = False;         
	print('\nNot able to converge!')
	theta = [0,0]        

    if(is_converged):	
        print('\n-----------Final Parameters-----------')
        print('Theta-0: {0:0.2f} \nTheta-1: {1:0.2f}'.format(*theta))
	print('\ny = '+str(theta[0])+' + '+str(theta[1])+'*x')

    draw_xymodel(x,y,alpha)
    
    zipped = zip(*t_list)
    draw_cost_function(zipped[0],j_list,'Theta-0',alpha)
    draw_cost_function(zipped[1],j_list,'Theta-1',alpha)
    
    x_arr = np.array([i for i in range(len(j_list))])
    draw_cost_function(x_arr,j_list,'Iterations',alpha)

