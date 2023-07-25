import numpy as np
import random
import math
import matplotlib.pyplot as plt

posp = 0
RadarEKF_first = True
dt = 0.05

A = np.eye(3)+dt*np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
Q = np.array([[0, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])    
R = 100
x = np.array([[0, 90, 1100]]).T
P = 1*np.eye(3)

def Hjacob(x):
    H = np.zeros((1,3))
    H[0][0] = x[0]/math.sqrt(x[0]**2 + x[2]**2)
    H[0][2] = x[2]/math.sqrt(x[0]**2 + x[2]**2)
    return H
def GetRadar(dt):
    global posp
    
    vel = 100 + 5*random.gauss(0, 1)
    alt = 1000 + 10*random.gauss(0, 1)
    
    pos = posp + vel*dt
    v = pos * 0.05 * random.gauss(0, 1)
    r = math.sqrt(pos**2 + alt**2) + v
    
    posp = pos
    return r
def RadarEKF(z): 
    global x, P, A, Q, R 
    H = Hjacob(x)
    
    xp = np.matmul(A, x)
    Pp = np.matmul(np.matmul(A, P), np.transpose(A)) + Q
    K = np.matmul(Pp, np.transpose(H))/(np.matmul(np.matmul(H, Pp), np.transpose(H)) + R)
    x = xp + K*(z - hx(xp))
    P = Pp - np.matmul(np.matmul(K, H), Pp)
    
    return x[0], x[1], x[2]
    



def hx(xp):
    return math.sqrt(xp[0]**2 + xp[2]**2)
    

dt = 0.05
t = list(range(0,1000))
N_samples = len(t)

Xsaved = np.zeros((N_samples, 3))
Zsaved = np.zeros((N_samples, 1))
EKFdistance = np.zeros((N_samples, 1))

for i in range(N_samples):
    r = GetRadar(dt)
    
    pos, vel, alt = RadarEKF(r)
    
    EKFdistance[i] = math.sqrt(pos**2 + alt**2)
    Xsaved[i][0] = pos
    Xsaved[i][1] = vel
    Xsaved[i][2] = alt
    Zsaved[i] = r

fig = plt.figure(figsize=(20,13))
ax1 = fig.add_subplot(411)
ax1.plot(t, Zsaved, 'bo', markersize = 2, label='z', linewidth=0.4)
ax1.plot(t, EKFdistance, 'ro', markersize = 2, linestyle='solid', label='EKFdistance', linewidth=0.4)
ax1.legend(loc='upper left')

ax2 = fig.add_subplot(412)
ax2.plot(t, [row[0] for row in Xsaved], 'ro', markersize = 2, linestyle='solid', label='pos', linewidth=0.4)
ax2.legend(loc='upper left')

ax3 = fig.add_subplot(413)
ax3.plot(t, [row[1] for row in Xsaved], 'ro', markersize = 2, linestyle='solid', label='vel', linewidth=0.4)
ax3.legend(loc='upper left')

ax4 = fig.add_subplot(414)
ax4.plot(t, [row[2] for row in Xsaved], 'ro', markersize = 2, linestyle='solid', label='alt', linewidth=0.4)
ax4.legend(loc='upper left')


plt.tight_layout()  # To prevent overlapping of subplots
plt.show()

    
    