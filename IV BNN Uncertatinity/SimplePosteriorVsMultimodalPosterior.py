import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(-3, 3, 500)
simplified_posterior = np.exp(-0.5 * x**2)
multimodal_posterior = np.exp(-0.5 * (x - 1)**2) + np.exp(-0.5 * (x + 1)**2)

# Plot the posteriors
plt.figure(figsize=(8, 6))
plt.plot(x, simplified_posterior, label="Simplified Posterior (Unimodal)", color="blue", lw=2)
plt.plot(x, multimodal_posterior, label="True Multimodal Posterior", color="red", lw=2)
plt.fill_between(x, simplified_posterior, color="blue", alpha=0.2)
plt.fill_between(x, multimodal_posterior, color="red", alpha=0.2)

# Add title, labels, and grid
plt.title("Simplified Posterior vs. Multimodal Posterior", fontsize=16)
plt.xlabel("Parameter Space", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.grid(alpha=0.3)

# Place legend in the bottom-middle inside the plot area
plt.legend(
    fontsize=12,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.05),  # Positioned slightly above the x-axis
    frameon=True,  # Add a box around the legend
    fancybox=True,  # Rounded edges for the box
    shadow=False  # No shadow for simplicity
)

plt.show()
