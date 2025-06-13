import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 500)
true_posterior = np.exp(-0.5 * x**2)
approximate_posterior = np.exp(-0.5 * (x - 1)**2)

plt.figure(figsize=(8, 6))
plt.plot(x, true_posterior, label="True Posterior", color="red", lw=2)
plt.plot(x, approximate_posterior, label="Approximate Posterior", color="blue", lw=2)
plt.fill_between(
    x, true_posterior, approximate_posterior, where=approximate_posterior > true_posterior,
    color="green", alpha=0.3, label="Under-penalized Region"
)
plt.title("KL Divergence Penalty Asymmetry", fontsize=16)
plt.xlabel("Parameter Space", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
