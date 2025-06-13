import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(-3, 3, 500)
mean_field = np.exp(-0.5 * x**2)  # Mean-field approximation (unimodal Gaussian)
true_posterior = np.exp(-0.5 * (x - 1)**2) + np.exp(-0.5 * (x + 1)**2)  # Multimodal posterior

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(x, mean_field, label="Mean-field Approximation", color="blue", lw=2)
plt.plot(x, true_posterior, label="True Posterior (Multimodal)", color="red", lw=2)
plt.fill_between(x, mean_field, color="blue", alpha=0.2)  # Shading for mean-field
plt.fill_between(x, true_posterior, color="red", alpha=0.2)  # Shading for true posterior

# Add title, labels, and grid
plt.title("Mean-field Approximation vs. True Posterior", fontsize=16)
plt.xlabel("Parameter Space", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.grid(alpha=0.3)

# Place legend in the bottom-middle inside the plot area
plt.legend(
    fontsize=12,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.05),  # Positioned slightly above the x-axis
    frameon=True,  # Add a box around the legend
    fancybox=True,  # Rounded corners for the legend box
    shadow=False  # No shadow for simplicity
)

# Show plot
plt.show()
