# Applying Kalman Filter to velocity estimation

import random
import numpy as np
import matplotlib.pyplot as plt

def generate_random_data(mean, std, num_data_points):
    data = []
    for i in range(num_data_points):
        data.append(i*8+random.gauss(mean, std))
    return data

z = np.array(generate_random_data(0, 10, 100))
measured_v = []
for i in range(100):
    measured_v.append(z[i]-i*8+8)

dt = 0.1
x = np.array([[1, 1]]) # position and velocity
x_ = np.array([[1, 1]]) 
P = np.array([1 * np.eye(2)])
P_ = np.array([1 * np.eye(2)])
A = np.array([[1 ,dt], [0, 1]])
H = np.array([1, 0])
H = np.array([0, 1])
Q = np.array([[1, 0], [0, 3]])
R = 10
K = np.array([1])

for i in range(1, 100):
    # prediction
    x_ = np.append(x_, [np.matmul(A, x[i-1])], axis=0)
    P_ =np.append(P_, [np.matmul(np.matmul(A, P[i-1]), np.transpose(A)) + Q], axis=0)
    
    # calculate Kalman gain
    # K = np.append(K, P_[i]/(P_[i] + R))
    K = np.append(K, np.matmul(P_[i], np.transpose(H))/(np.matmul(np.matmul(H, P_[i]), np.transpose(H)) + R))
    
    # calculate posterior
    # x = np.append(x, [x_[i] + K[i]*(z[i] - np.matmul(H, x_[i]))], axis=0)
    x = np.append(x, [x_[i] + K[i]*(measured_v[i] - np.matmul(H, x_[i]))], axis=0)
    
    # calculate posterior covariance
    P = np.append(P, [(1 - K[i])*P_[i]], axis=0)

    

plt.figure(figsize=(20,7))
# plt.plot(z, 'bo', markersize = 2, label='z', linewidth=0.4)
plt.plot(measured_v, 'bo', markersize = 2, label='measured_v', linewidth=0.4)
plt.plot([row[0] for row in x], 'ro', markersize = 2, linestyle='solid', label='v', linewidth=0.4)
plt.legend(loc='upper left')
plt.show()