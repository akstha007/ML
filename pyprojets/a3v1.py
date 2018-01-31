import matplotlib.pyplot as plt
from numpy import ones


x =    [2.0658746e+00,
   2.3684087e+00,
   2.5399929e+00,
   2.5420804e+00,
   2.5490790e+00,
   2.7866882e+00,
   2.9116825e+00,
   3.0356270e+00,
   3.1146696e+00,
   3.1582389e+00,
   3.3275944e+00,
   3.3793165e+00,
   3.4122006e+00,
   3.4215823e+00,
   3.5315732e+00,
   3.6393002e+00,
   3.6732537e+00,
   3.9256462e+00,
   4.0498646e+00,
   4.2483348e+00,
   4.3440052e+00,
   4.3826531e+00,
   4.4230602e+00,
   4.6102443e+00,
   4.6881183e+00,
   4.9777333e+00,
   5.0359967e+00,
   5.0684536e+00,
   5.4161491e+00,
   5.4395623e+00,
   5.4563207e+00,
   5.5698458e+00,
   5.6015729e+00,
   5.6877617e+00,
   5.7215602e+00,
   5.8538914e+00,
   6.1978026e+00,
   6.3510941e+00,
   6.4797033e+00,
   6.7383791e+00,
   6.8637686e+00,
   7.0223387e+00,
   7.0782373e+00,
   7.1514232e+00,
   7.4664023e+00,
   7.5973874e+00,
   7.7440717e+00,
   7.7729662e+00,
   7.8264514e+00,
   7.9306356e+00]

y =    [7.7918926e-01,
   9.1596757e-01,
   9.0538354e-01,
   9.0566138e-01,
   9.3898890e-01,
   9.6684740e-01,
   9.6436824e-01,
   9.1445939e-01,
   9.3933944e-01,
   9.6074971e-01,
   8.9837094e-01,
   9.1209739e-01,
   9.4238499e-01,
   9.6624578e-01,
   1.0526500e+00,
   1.0143791e+00,
   9.5969426e-01,
   9.6853716e-01,
   1.0766065e+00,
   1.1454978e+00,
   1.0340625e+00,
   1.0070009e+00,
   9.6683648e-01,
   1.0895919e+00,
   1.0634462e+00,
   1.1237239e+00,
   1.0323374e+00,
   1.0874452e+00,
   1.0702988e+00,
   1.1606493e+00,
   1.0778037e+00,
   1.1069758e+00,
   1.0971875e+00,
   1.1648603e+00,
   1.1411796e+00,
   1.0844156e+00,
   1.1252493e+00,
   1.1168341e+00,
   1.1970789e+00,
   1.2069462e+00,
   1.1251046e+00,
   1.1235672e+00,
   1.2132829e+00,
   1.2522652e+00,
   1.2497065e+00,
   1.1799706e+00,
   1.1897299e+00,
   1.3029934e+00,
   1.2601134e+00,
   1.2562267e+00]

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
theta = [0.0,0.0]
iterations = 1500
alpha = 0.07

j_list = []
t_list = []
x_arr = [i for i in range(iterations)]


m = len(y)
n = len(theta)
it = ones(shape=(m, 2))
it[:, 1] = x

for i in range(iterations):
    j_theta = compute_cost(x,y,n,m,theta)
    j_list.append(j_theta)
    theta = gradient_descent(it,y,m,theta,alpha)
    t_list.append(theta)    

plt.title('Simple Linear Regression')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.scatter(x,y,label="Actual Value",color="g",marker="o",s=1)

approx_value = [hypothesis(x_row,n,theta) for x_row in it]
plt.plot(x, approx_value, label="Predicted Value", color='r', linewidth = 1)

plt.legend()
plt.show()

#print(hypothesis([1,7],2,theta))

zipped = zip(*t_list)
#plt.plot(x_arr, zipped[0], label="Theta_0 Value", color='m', linewidth = 1)
#plt.plot(x_arr, zipped[1], label="Theta_1 Value", color='b', linewidth = 1)

plt.title('Gradient Descent')
plt.xlabel('Theta-0')
plt.ylabel('Theta-1')
#plt.plot(zipped[0], zipped[1], label="Theta Values", color='g', linewidth = 1)
plt.scatter(zipped[0], zipped[1],label="Theta Values",color="b",marker="o",s=1)
plt.legend()
plt.show()

plt.title('Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost function')
#plt.plot(x_arr, j_list, label="J-Theta", color='m', linewidth = 1)
plt.scatter(x_arr, j_list,label="J-Theta",color="m",marker="o",s=1)
plt.legend()
plt.show()

