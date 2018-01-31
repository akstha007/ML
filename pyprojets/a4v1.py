import matplotlib.pyplot as plt
from numpy import ones, zeros
import numpy as np


x = np.array([[2.1040000e+03,3.0000000e+00],
   [1.6000000e+03,3.0000000e+00],
   [2.4000000e+03,3.0000000e+00],
   [1.4160000e+03,2.0000000e+00],
   [3.0000000e+03,4.0000000e+00],
   [1.9850000e+03,4.0000000e+00],
   [1.5340000e+03,3.0000000e+00],
   [1.4270000e+03,3.0000000e+00],
   [1.3800000e+03,3.0000000e+00],
   [1.4940000e+03,3.0000000e+00],
   [1.9400000e+03,4.0000000e+00],
   [2.0000000e+03,3.0000000e+00],
   [1.8900000e+03,3.0000000e+00],
   [4.4780000e+03,5.0000000e+00],
   [1.2680000e+03,3.0000000e+00],
   [2.3000000e+03,4.0000000e+00],
   [1.3200000e+03,2.0000000e+00],
   [1.2360000e+03,3.0000000e+00],
   [2.6090000e+03,4.0000000e+00],
   [3.0310000e+03,4.0000000e+00],
   [1.7670000e+03,3.0000000e+00],
   [1.8880000e+03,2.0000000e+00],
   [1.6040000e+03,3.0000000e+00],
   [1.9620000e+03,4.0000000e+00],
   [3.8900000e+03,3.0000000e+00],
   [1.1000000e+03,3.0000000e+00],
   [1.4580000e+03,3.0000000e+00],
   [2.5260000e+03,3.0000000e+00],
   [2.2000000e+03,3.0000000e+00],
   [2.6370000e+03,3.0000000e+00],
   [1.8390000e+03,2.0000000e+00],
   [1.0000000e+03,1.0000000e+00],
   [2.0400000e+03,4.0000000e+00],
   [3.1370000e+03,3.0000000e+00],
   [1.8110000e+03,4.0000000e+00],
   [1.4370000e+03,3.0000000e+00],
   [1.2390000e+03,3.0000000e+00],
   [2.1320000e+03,4.0000000e+00],
   [4.2150000e+03,4.0000000e+00],
   [2.1620000e+03,4.0000000e+00],
   [1.6640000e+03,2.0000000e+00],
   [2.2380000e+03,3.0000000e+00],
   [2.5670000e+03,4.0000000e+00],
   [1.2000000e+03,3.0000000e+00],
   [8.5200000e+02,2.0000000e+00],
   [1.8520000e+03,4.0000000e+00],
   [1.2030000e+03,3.0000000e+00]])
y = np.array([3.9990000e+05,
   3.2990000e+05,
   3.6900000e+05,
   2.3200000e+05,
   5.3990000e+05,
   2.9990000e+05,
   3.1490000e+05,
   1.9899900e+05,
   2.1200000e+05,
   2.4250000e+05,
   2.3999900e+05,
   3.4700000e+05,
   3.2999900e+05,
   6.9990000e+05,
   2.5990000e+05,
   4.4990000e+05,
   2.9990000e+05,
   1.9990000e+05,
   4.9999800e+05,
   5.9900000e+05,
   2.5290000e+05,
   2.5500000e+05,
   2.4290000e+05,
   2.5990000e+05,
   5.7390000e+05,
   2.4990000e+05,
   4.6450000e+05,
   4.6900000e+05,
   4.7500000e+05,
   2.9990000e+05,
   3.4990000e+05,
   1.6990000e+05,
   3.1490000e+05,
   5.7990000e+05,
   2.8590000e+05,
   2.4990000e+05,
   2.2990000e+05,
   3.4500000e+05,
   5.4900000e+05,
   2.8700000e+05,
   3.6850000e+05,
   3.2990000e+05,
   3.1400000e+05,
   2.9900000e+05,
   1.7990000e+05,
   2.9990000e+05,
   2.3950000e+05])

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


def draw_cost(x_arr, j_list):
    plt.title('Cost Function')
    plt.xlabel('Iterations')
    plt.ylabel('Cost function')
    plt.plot(x_arr, j_list, label="J-Theta", color='m', linewidth = 1)
    #plt.scatter(x_arr, j_list,label="J-Theta",color="m",marker="o",s=1)
    plt.legend()
    plt.show()

m = len(y)
n = x.shape[1]+1
#assume theta0 and theta1 values initially
theta = np.array([0 for i in range(n)])
#theta = np.zeros((n,1))
print('Intial parameters')
print(theta)

iterations = 50
alpha = 0.01
j_list = []
t_list = []

x_arr = [i for i in range(iterations)]
it = np.ones((m,n))
it[:,1:] = x

t_list, j_list = without_scaling(it,y,n,m,theta,alpha)
#t_list, j_list = with_scaling(it,y,n,m,theta,alpha)

draw_xymodel(x,y)
draw_cost(x_arr,j_list)


#zipped = zip(*t_list)
#plt.plot(x_arr, zipped[0], label="Theta_0 Value", color='m', linewidth = 1)
#plt.plot(x_arr, zipped[1], label="Theta_1 Value", color='b', linewidth = 1)



