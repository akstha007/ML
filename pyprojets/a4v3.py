import matplotlib.pyplot as plt
from numpy import ones, zeros
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

def with_scaling(x,y,n,m,theta,alpha):
    '''
	Scaling the parameters: Dividing by features by the range
    '''
    t_list = []
    j_list = []
    range_i = x.max(axis=0)-x.min(axis=0)
    
    scale_x = x
    for i in range(1,n,1):
        scale_x[:,i] = x[:,i]/range_i[i]
    
    for i in range(iterations):    
        theta = gradient_descent(scale_x,y,n,m,theta,alpha)
        t_list.append(theta)
        j_theta = compute_cost(scale_x,y,n,m,theta)
        j_list.append(j_theta)
    
    print('\nFinal parameters with Scaling')
    print(theta)
    
    return t_list,j_list


def without_scaling(x,y,n,m,theta,alpha):
    '''
	Without scaling
    '''
    for i in range(iterations):    
        theta = gradient_descent(x,y,n,m,theta,alpha)
        t_list.append(theta)
        j_theta = compute_cost(x,y,n,m,theta)
        j_list.append(j_theta)
    
    print('\nFinal parameters without Scaling')
    print(theta)
        
    return t_list,j_list

def draw_xymodel(x,y):
    zipped = zip(*x)
    plt.title('Multiple Linear Regression without Scaling')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.scatter(zipped[0],y,label="X0",color="g",marker="o",s=3)
    plt.scatter(zipped[1],y,label="X1",color="b",marker="*",s=3)
    plt.legend()
    plt.show()


def draw_cost_function(x_arr, j_list):
    plt.title('Cost Function')
    plt.xlabel('Iterations')
    plt.ylabel('Cost function')
    plt.plot(x_arr, j_list, label="J-Theta", color='m', linewidth = 1)
    #plt.scatter(x_arr, j_list,label="J-Theta",color="m",marker="o",s=1)
    plt.legend()
    plt.show()

x = np.array([[2.0,3.0,4.0,4.0,3.0,7.0,5.0,3.0,2.0],
     [70.0,30.0,80.0,20.0,50.0,10.0,50.0,90.0,20.0]])

y = np.array([79.0,41.5,97.5,36.1,63.2,39.5,69.8,103.5,29.5])

m = len(y)
n = x.shape[0]+1
#assume theta0 and theta1 values initially
theta = np.array([0 for i in range(n)])
#theta = np.zeros((n,1))
print('Intial parameters')
print(theta)

iterations = 1500
alpha = 1.00
'''
some alpha values:
without scaling =0.000684,0.0001(fast convergence)
with scaling = 1.00(fast convergence)
'''
j_list = []
t_list = []

x_arr = [i for i in range(iterations)]
it = np.ones((m,n))
it[:,1:] = np.array(zip(*x))

#t_list, j_list = without_scaling(it,y,n,m,theta,alpha)
t_list, j_list = with_scaling(it,y,n,m,theta,alpha)

draw_xymodel(np.array(zip(*x)),y)
draw_cost_function(x_arr,j_list)

