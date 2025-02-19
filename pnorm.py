"""
Graphs for the ilustration of the p-norms: L0, L1, and L2.
This is not a matheamtical example, just a mock-up.

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia

Date:
    January 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from functions import save_image


# Parameters for the orange line (representing the solution space).
def solution_space(x):
    return -0.5 * x + 1.5  

# Set some prefix.
plot_kws = dict(linewidth=1, alpha=0.7) # plot keywords.
ax_lim   = 2.5                          # The lims for the axes

# Create the data.
x_vals = np.linspace(-ax_lim, ax_lim, 2)
y_vals = solution_space(x_vals)

# Define the figure and axes.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Set up grid for all axes
for ax in axes:
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.set(
        xlim=(-ax_lim, ax_lim), ylim=(-ax_lim, ax_lim), 
        xlabel=f"$s_1$", ylabel=f"$s_2$"
        )
    # Plot the solution line.
    ax.plot(x_vals, y_vals, color="red", label="Family of solutions")

# -----
# Plot for L2-norm (Left figure)
# Intersects the solution with the minimum radious 

# The intersection point (for the given line equation).
x_l2 = 3/5
r_l2 = 1.34

#radious_l2 = [r_l2-1.25, r_l2-1.0, r_l2-0.75, r_l2-0.5, r_l2-0.25, r_l2]
radious_l2 = [r_l2-0.34*3, r_l2-0.34*2, r_l2-0.34, r_l2]

for r in radious_l2:
    circle = plt.Circle((0, 0), r, color="blue", fill=False)
    axes[0].add_artist(circle)
axes[0].scatter([x_l2], [solution_space(x_l2)], color="green", label=f"Solution ({x_l2}, {r_l2})", zorder=5)
axes[0].set_title("L2-Norm constraint")
axes[0].legend()

# -----
# Plot for L0-norm (Middle figure)

# The intersection point.
x_l0 = 0
r_l0 = 1.5

axes[1].plot([-1.5, 1.5], [0, 0], color="blue")
axes[1].plot([0, 0], [-1.5, 1.5], color="blue")
axes[1].scatter([x_l0], [solution_space(x_l0)], color="green", label=f"Solution ({x_l0}, {r_l0})", zorder=5)
axes[1].set_title("L0-Norm constraint")
axes[1].legend()

# -----
# Plot for L1-norm (Right figure)

radious_l1 = [r_l0-0.34*3, r_l0-0.34*2, r_l0-0.34, r_l0]
for r in radious_l1:
    diamond = np.array([[r, 0], [0, r], [-r, 0], [0, -r], [r, 0]])
    axes[2].plot(diamond[:, 0], diamond[:, 1], color="blue")
axes[2].scatter([x_l0], [solution_space(x_l0)], color="green", label=f"Solution ({x_l0}, {r_l0})", zorder=5)
axes[2].set_title("L1-Norm constraint")
axes[2].legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Save the image
# save_image(fig, "p-norm")

# Fin :)