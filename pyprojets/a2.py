import matplotlib.pyplot as plt

#Hypothesis function
def hypothesis(n, m, x, theta):
    h_theta = sum([sum([theta[i]*x[i-1][j] for j in range(1,m,1)]) for i in range(n)]) + theta[0]    
    return h_theta


#calculate cost funtion
def compute_cost(x,y,n,m,theta):
#    for i in range(m):
 #       h_theta.append(sum(theta[j+1]*x[i][j] for j in range(n)))

    h_theta = [sum([theta[j+1] * x[i][j] for j in range(n)]) for i in range(m)]        
    j_theta = (sum([h_theta[i]-y[i] for i in range(m)])**2)/(2*m)
    return j_theta, h_theta

#calculate new values of thetas using gradient descent
def gradient_descent(theta, alpha, j_theta):
    theta[0] = theta[0] - alpha*j_theta
    theta[1] = theta[1] - alpha*j_theta
    return theta

x = [[2.0,3.0,4.0,4.0,3.0,7.0,5.0,3.0,2.0],
     [70.0,30.0,80.0,20.0,50.0,10.0,50.0,90.0,20.0]]

y = [79.0,41.5,97.5,36.1,63.2,39.5,69.8,103.5,29.5]

plt.scatter(x[0],y,label="X1 Value",color="g",marker="o",s=30)
plt.scatter(x[1],y,label="X2 Value",color="b",marker="o",s=30)
plt.xlabel('x - axis')
plt.ylabel('y - axis')

#assume theta0 and theta1 values initially
m = len(y)
n = len(x)
theta = [3.0,1.0,2.0]

h_theta = hypothesis(n,m,x,theta)
print(h_theta)
j_theta, h_theta = compute_cost(x,y,n,m, h_theta)
print(j_theta, h_theta)

iterations = 1000
alpha = 0.01
tolerance = 0.1
is_optimal = False
j_list = []
t_list = []

#initial value of parameters
print('Initial Values: \nTheta0: {0:0.2f}, Theta1: {1:0.2f}'.format(*theta))

plt.title('Multiple Linear Regression')
plt.legend()
plt.show()

