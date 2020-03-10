import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from Driver import Driver
from Quad_Driver import QuadDriver
from NN_Driver import NN_driver

#CONSTANTS
SEED = 520
np.random.seed(SEED)

#number of parking spaces (without garage)
N = 10
myN = np.linspace(1,N,N,dtype=int)

# probability of space being free
x = np.linspace(0, 2*np.pi, N+1)
pf = 0.8
#print('Probability of every spot being free is:\n', pf)

#cost of every spot
c = (np.sin(x*2)+1)/2
c[N] = 2
print('Cost of every spot is:\n', c)


# space actually free or not. Here all spots are free
f = np.zeros(N,int)
f[pf>=0.3]=1
f[np.argmin(c)] = 1
print('Whether every spot is actually free or not:\n', f)

exact_driver = Driver(pf,f,c,N)
#learned_policy = exact_driver.policy_iter()

qdriver = QuadDriver(pf,f,c,N)
learned_policy = qdriver.policy_iter()


nndriver = NN_driver(pf,f,c,N)
#learned_policy = nndriver.policy_iter()


n = np.array([*myN,N+1])
nn = n[:-1]
cc = c[:-1]
# Plot the cost function
fig = plt.figure(1)
plt.plot(n,c)
plt.plot(nn[learned_policy=='park'],cc[learned_policy=='park'],'ro')
plt.title('Actual Cost function, chosen parking spot is red')
plt.xlabel('Parking Spots')
plt.ylabel('Cost')
plt.show()


