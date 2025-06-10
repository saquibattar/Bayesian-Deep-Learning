import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
N = 20
x = np.linspace(-1, 1, N)
true_w = 2.5
y = true_w * x + np.random.normal(0, 0.2, N)

# Bayesian update
tau2 = 1.0
sigma2 = 0.2 ** 2
sigma_N2 = 1 / (1/tau2 + np.sum(x**2) / sigma2)
mu_N = sigma_N2 * np.sum(x * y) / sigma2

# Plotting posterior
w_vals = np.linspace(0, 5, 200)
posterior = (1 / np.sqrt(2 * np.pi * sigma_N2)) * np.exp(-(w_vals - mu_N)**2 / (2 * sigma_N2))
plt.plot(w_vals, posterior, label='Posterior')
plt.axvline(x=true_w, linestyle='--', label='True w', color='r')
plt.title('Posterior over Weight')
plt.legend()
plt.xlabel('Weight')
plt.ylabel('Density')
plt.show()
