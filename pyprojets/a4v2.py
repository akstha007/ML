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
    plt.scatter(zipped[0],y,label="Actual Value",color="g",marker="o",s=1)
    plt.scatter(zipped[1],y,label="Actual Value",color="b",marker="*",s=1)
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

iterations = 50
alpha = 0.1
j_list = []
t_list = []

x_arr = [i for i in range(iterations)]
it = np.ones((m,n))
#print(np.array(zip(*x)))
it[:,1:] = np.array(zip(*x))

t_list, j_list = without_scaling(it,y,n,m,theta,alpha)
#t_list, j_list = with_scaling(it,y,n,m,theta,alpha)

draw_xymodel(np.array(zip(*x)),y)
draw_cost_function(x_arr,j_list)


#zipped = zip(*t_list)
#plt.plot(x_arr, zipped[0], label="Theta_0 Value", color='m', linewidth = 1)
#plt.plot(x_arr, zipped[1], label="Theta_1 Value", color='b', linewidth = 1)



