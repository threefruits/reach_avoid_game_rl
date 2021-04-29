import numpy as np
import matplotlib.pyplot as plt

a=np.load("record.npy")
# y = np.linspace(-5, 5, 100)\
time = 120
print(a)
plt.vlines(1, -1, 1, colors = "r", linestyles = "dashed")

plt.plot(a[:time,0],a[:time,1],label = 'attacker')
plt.plot(a[:time,3],a[:time,4],label = "defender")
plt.xlabel('x/(m)')
plt.ylabel('y/(m)')
plt.legend()

plt.xlim((-1.5, 1.5))
plt.ylim((-1, 1))
plt.show()