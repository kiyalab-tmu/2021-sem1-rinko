import matplotlib.pyplot as plt

a = [1,2 ,3 ,4, 5, 6]
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(a)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(a)
plt.show()
