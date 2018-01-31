import matplotlib.pyplot as plt

#calculate cost funtion
def compute_cost(x,y,theta):
    h_theta = [theta[0] + theta[1] * xi for xi in x]
        
    m = len(y)
    j_theta = (sum([h_theta[i]-y[i] for i in range(m)])**2)/(2*m)
    return j_theta, h_theta

#calculate new values of thetas using gradient descent
def gradient_descent(theta, alpha, j_theta):
    theta[0] = theta[0] - alpha*j_theta
    theta[1] = theta[1] - alpha*j_theta
    return theta

x = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
y = [5.1,6.1,6.9,7.8,9.2,9.9,11.5,12.0,12.8]

#plt.scatter(x,y,label="Actual Value",color="g",marker="o",s=30)
plt.xlabel('x - axis')
plt.ylabel('y - axis')

#assume theta0 and theta1 values initially
theta = [3.0,3.0]
iterations = 100
alpha = 0.01
tolerance = 0.1
is_optimal = False
j_list = []
t_list = []
x_arr = []

#initial value of parameters
print('Initial Values: \nTheta0: {0:0.2f}, Theta1: {1:0.2f}'.format(*theta))

for i in range(iterations):
    j_theta, h_theta = compute_cost(x,y,theta)

    if(max(h_theta)>10):
	alpha = alpha/10
	theta = gradient_descent(theta, alpha, j_theta)
        print(alpha,theta)
	j_theta, h_theta = compute_cost(x,y,theta)  
        #print(j_theta, h_theta)

    if(min(h_theta)<-10):
	alpha = alpha*10
	theta = gradient_descent(theta, alpha, j_theta)
        print(alpha,theta)
	j_theta, h_theta = compute_cost(x,y,theta)  
        #print(j_theta, h_theta)
    
    j_list.append(j_theta)
    theta = gradient_descent(theta,alpha, j_theta)
    t_list.append(theta)
    x_arr.append(i)
    if j_theta<=tolerance:
	is_optimal = True
        break;

if(is_optimal):
    print('Final Values: \nTheta0: {0:0.2f}, Theta1: {1:0.2f}'.format(*theta))

    zipped = zip(*t_list)
    print(x_arr,zipped[0])
    
    plt.plot(x_arr, zipped[0], color="r")
    plt.plot(x_arr, zipped[1], color="b")
    #print(len(zipped[0]),len(j_list))
    #print(list(zipped))
    #print(j_list)

    h_theta = [theta[0] + theta[1] * xi for xi in x]
    #plt.plot(x, h_theta, label="Predicted Value", color='m', linestyle='dashed', linewidth = 1,marker='*', markerfacecolor='r', markersize=6)

else:
    print("No optimal solution found.")

plt.title('Simple Linear Regression')
plt.legend()
plt.show()

