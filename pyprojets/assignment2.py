'''
------------------Assignment 1--------------------

Multiple Linear Regression
==========================

Submitted by: Ashok Kumar Shrestha
Date: Aug 29, 2017


'''

import matplotlib.pyplot as plt
import numpy as np

#Hypothesis function
def hypothesis(x_row,n,theta):
    return sum([theta[i]*x_row[i] for i in range(n)])

#calculate cost funtion
def compute_cost(x,y,n,m,theta):
    j_theta = sum([(hypothesis(x[i],n,theta)-y[i])**2 for i in range(m)])/(2*m)
    return j_theta

#calculate new values of thetas using gradient descent
def gradient_descent(x,y,n,m,theta,alpha):
    n_theta = np.zeros((n,1))
    for j in range(n):
        n_theta[j] = theta[j] - alpha*(sum([(hypothesis(x[i],n,theta)-y[i])*x[i][j] for i in range(m)]))/(m)

    return n_theta

#Feature scaling: Dividing each feature value by the range of the feature 
def with_scaling(x,y,n,m,theta,alpha):    
    t_list = []
    j_list = []
    is_converged = True;
    range_i = x.max(axis=0)-x.min(axis=0)
    
    scale_x = x
    for i in range(1,n,1):
        scale_x[:,i] = x[:,i]/range_i[i]
    try:
        for i in range(iterations):    
            theta = gradient_descent(scale_x,y,n,m,theta,alpha)
            t_list.append(theta)
            j_theta = compute_cost(scale_x,y,n,m,theta)
            j_list.append(j_theta)
    except FloatingPointError as e:
        #print(e)    
	is_converged = False;         
        theta = np.array([0 for i in range(n)])
	j_list.append(j_theta)
	print('\nNot able to converge!')
   
    if(is_converged):
        print('\n-----------Final Parameters with Scaling-----------') 	
        for i in range(n):
	    print('theta-'+str(i)+': '+str(*theta[i]))

	eqn = '\ny = ' 
	eqn += str(*theta[0])  
	for i in range(1,n,1):
	    eqn += ' + '+str(*theta[i]) + '*x'+str(i)
	print(eqn)
    
    return t_list,j_list

#Without scaling the predictor variables
def without_scaling(x,y,n,m,theta,alpha):
    is_converged = True;
    try:
        for i in range(iterations):    
            theta = gradient_descent(x,y,n,m,theta,alpha)
            t_list.append(theta)
            j_theta = compute_cost(x,y,n,m,theta)
            j_list.append(j_theta)
    except FloatingPointError as e:
        #print(e)  
	is_converged = False;           
        theta = np.array([0 for i in range(n)])
	j_list.append(j_theta)
	print('\nNot able to converge!')

    if(is_converged):
        print('\n-----------Final Parameters without Scaling-----------')
        for i in range(n):
	    print('theta-'+str(i)+': '+str(*theta[i]))
	
	eqn = '\ny = ' 
	eqn += str(*theta[0])
	for i in range(1,n,1):
	    eqn += ' + '+str(*theta[i]) + '*x'+str(i)
	print(eqn)
        
    return t_list,j_list

#Plot given data sets in graph
def draw_xymodel(x,y):
    zipped = zip(*x)
    plt.title('Multiple Linear Regression without Scaling')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.scatter(zipped[0],y,label="X0",color="g",marker="o",s=3)
    plt.scatter(zipped[1],y,label="X1",color="b",marker="*",s=3)
    plt.legend()
    plt.show()

#plot cost function in graph
def draw_cost_function(x_arr, j_list,x_lable):
    plt.title('Cost Function')
    plt.xlabel(x_lable)
    plt.ylabel('Cost function')
    plt.plot(x_arr, j_list, color='g', linewidth = 1)
    plt.scatter(x_arr, j_list, color="m", marker="o", s=2)
    plt.show()

if __name__ == '__main__':
    
    x = np.array([[2.0,3.0,4.0,4.0,3.0,7.0,5.0,3.0,2.0],
     [70.0,30.0,80.0,20.0,50.0,10.0,50.0,90.0,20.0]])

    y = np.array([79.0,41.5,97.5,36.1,63.2,39.5,69.8,103.5,29.5])
    
    np.seterr(all = 'raise')
    m = len(y)
    n = x.shape[0]+1
    iterations = 1500
    alpha = 0.000684
    
    #assume theta0 and theta1 values initially
    theta = np.array([0 for i in range(n)])
    
    print('Alpha: '+str(alpha))
    print('-----------Initial Parameters-----------')
    for i in range(n):
	print('theta-'+str(i)+': '+str(theta[i]))
    
    '''
    some alpha values:
    without scaling =0.000684,0.0001(fast convergence)
    with scaling = 1.00(fast convergence)
    '''
    j_list = []
    t_list = []    
    it = np.ones((m,n))
    it[:,1:] = np.array(zip(*x))

    t_list, j_list = without_scaling(it,y,n,m,theta,alpha)
    #t_list, j_list = with_scaling(it,y,n,m,theta,alpha)

    draw_xymodel(np.array(zip(*x)),y)
    
    theta_list = np.array(zip(*t_list))
    for i in range(n):
        draw_cost_function(theta_list[i],j_list,'Theta-'+str(i))

    x_arr = [i for i in range(len(j_list))]
    draw_cost_function(x_arr,j_list,'Iterations')

