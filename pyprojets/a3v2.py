import matplotlib.pyplot as plt
from numpy import ones


x = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
y = [5.1,6.1,6.9,7.8,9.2,9.9,11.5,12.0,12.8]

#Hypothesis function
def hypothesis(x_row,n,theta):
    return sum([theta[i]*x_row[i] for i in range(n)])

#calculate cost funtion
def compute_cost(x,y,n,m,theta):
    j_theta = sum([(theta[0]+theta[1]*x[1]-y[i])**2 for i in range(m)])/(2*m)
    return j_theta

#calculate new values of thetas using gradient descent
def gradient_descent(x,y,m,theta,alpha):
    n_theta = [0,0]
    n_theta[0] = theta[0] - alpha*(sum([(hypothesis(x[i],n,theta)-y[i])*x[i][0] for i in range(m)]))/(m)
    n_theta[1] = theta[1] - alpha*(sum([(hypothesis(x[i],n,theta)-y[i])*x[i][1] for i in range(m)]))/(m)
    return n_theta

#assume theta0 and theta1 values initially
theta = [5.0,0.0]
iterations = 1500
alpha = 0.07

j_list = []
t_list = []
x_arr = [i for i in range(iterations)]


m = len(y)
n = len(theta)
it = ones(shape=(m, 2))
it[:, 1] = x

print('Initial Parameters: \nTheta-0: {0:0.2f}, Theta-1: {1:0.2f}'.format(*theta))

for i in range(iterations):
    j_theta = compute_cost(x,y,n,m,theta)
    j_list.append(j_theta)
    theta = gradient_descent(it,y,m,theta,alpha)
    t_list.append(theta)    

print('Final Parameters: \nTheta-0: {0:0.2f}, Theta-1: {1:0.2f}'.format(*theta))

plt.title('Simple Linear Regression')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.scatter(x,y,label="Actual Value",color="g",marker="o",s=1)

approx_value = [hypothesis(x_row,n,theta) for x_row in it]
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

