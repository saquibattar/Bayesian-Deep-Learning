import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Draw the plate for repeated process
plate = Rectangle((1, 1), 6, 3, edgecolor='black', facecolor='none', lw=2)
ax.add_patch(plate)
ax.text(6.8, 4, r'$N$', fontsize=12)

# Draw circles for variables
variables = {
    "w": (4, 5),   # w (weights)
    "x_i": (2.5, 3),  # x_i (inputs)
    "y_i": (5.5, 3),  # y_i (outputs)
}
for var, (x, y) in variables.items():
    circle = Ellipse((x, y), width=1.2, height=1.2, edgecolor='black', facecolor='white', lw=2)
    ax.add_patch(circle)
    ax.text(x, y, rf"${var}$", fontsize=14, ha='center', va='center')

# Draw arrows for dependencies
arrows = [
    ("w", "x_i"),
    ("w", "y_i"),
    ("x_i", "y_i"),
]
for start, end in arrows:
    x_start, y_start = variables[start]
    x_end, y_end = variables[end]
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=0.6))

# Set limits and aspect
ax.set_xlim(0, 8)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')

# Add description text
ax.text(3.5, 0.5, "Plate indicates repetition for N data points", fontsize=12, ha='center')

# Save or display the plot
plt.tight_layout()
plt.show()