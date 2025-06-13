import matplotlib.pyplot as plt
import numpy as np

# Define data
x = np.linspace(-5, 5, 500)
prior = np.exp(-0.5 * (x / 1.5)**2)  # Prior: Gaussian
likelihood = np.exp(-0.5 * ((x - 1) / 1.0)**2)  # Likelihood: Offset Gaussian
posterior = prior * likelihood  # Posterior: Unnormalized product
posterior /= np.max(posterior)  # Normalize for visualization

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, prior, label="Prior (P(w))", color="blue")
plt.plot(x, likelihood, label="Likelihood (P(D|w))", color="green")
plt.plot(x, posterior, label="Posterior (P(w|D))", color="red")
plt.fill_between(x, 0, prior, color="blue", alpha=0.2)
plt.fill_between(x, 0, likelihood, color="green", alpha=0.2)
plt.fill_between(x, 0, posterior, color="red", alpha=0.2)

# Annotations and labels
plt.title("Bayesian Inference: Prior, Likelihood, and Posterior", fontsize=16)
plt.xlabel("Parameter (w)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Save and show the plot
output_path = "D:/IT/2nd Semester/Computational Intelligence/GIT Branch/Bayesian-Deep-Learning/IV BNN Uncertatinity/bayes_theorem_diagram.png"
plt.savefig(output_path, bbox_inches="tight")
plt.close()

output_path
