from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

mu = 0
variance = 1
sigma = np.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
x = np.arange(-10,10,0.001)
ax = plt.plot(x, norm.pdf(x, mu, sigma), label='T0')

mu += 1
variance += 2
sigma = np.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
x = np.arange(-10,10,0.001)
plt.plot(x, norm.pdf(x, mu, sigma), linestyle='--', color='r', label='T1 (with ST.DEV)')
plt.legend(fontsize='x-large')
plt.xticks([])
plt.yticks([])
plt.show()

del mu, variance, sigma, x, ax