import random

def generate_random_data(mean, std, num_data_points):
    data = []
    for i in range(num_data_points):
        data.append(random.gauss(mean, std))
    return data

x = [1]
x_ = [1]
P = [1]
P_ = [1]
A = 1

Q = 1
K = [0]
R = 10
z = generate_random_data(30, 10, 1000)
for i in range(1, 1000):
    # prediction
    x_.append(A*x[i-1])
    P_.append(A*P[i-1]*A + Q)
    
    # calculate Kalman gain
    K.append(P_[i]/(P_[i] + R))
    
    # calculate posterior
    x.append(x_[i] + K[i]*(z[i] - x_[i]))
    
    # calculate posterior covariance
    P.append((1 - K[i])*P_[i])
    
import matplotlib.pyplot as plt
plt.figure(figsize=(20,7))
plt.plot(z, 'bo', markersize = 2, color='blue', linestyle='dashed', label='z', linewidth=0.4)
plt.plot(x, 'ro', markersize = 2, color='red', linestyle='dashed', label='x', linewidth=0.4)
plt.show()