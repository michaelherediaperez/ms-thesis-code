"""
This is a mockup of the pareto front in the SINDy context.

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia
    
Date: 
    january 2025
"""

import numpy as np 
import matplotlib.pyplot as plt

from functions import save_image


# Define the domain
x = np.linspace(0, 5, 200)

# Smooth factors.
c = 1
k = 0.03

# Bifurcation point
x_mid = 1.5

# Only show bifurcation after this point
bifurcation_mask = x >= x_mid  

# The curves.
y_solid = lambda x : c*np.exp(-x)
y_dashd = lambda x : c*np.exp(-x) + k*(x - x_mid)**2 * bifurcation_mask


# Create the plot
fig = plt.figure(figsize=(9, 3))

# Plot the curves.
plt.plot(x, y_solid(x), "b-")
plt.plot(x, y_dashd(x), "b--")

# Plot the SINDy approximation area.
plt.scatter(1.8, y_solid(1.8), s=2200, color='lightblue', alpha=0.5, edgecolors='none')

# Axes labels.
plt.xlabel("Complexity $1/\lambda$\n(# terms) ")
plt.ylabel("Cross-validation\n(Error)")

# Text labels.
plt.plot(0, y_solid(0), 'b.', zorder=3)
plt.text(0, y_solid(0), "(Underfit)\n$\lambda = \infty$", color="b", verticalalignment="bottom", horizontalalignment="center")

plt.plot(1.8, y_solid(1.8), 'b.', zorder=3)
plt.text(1.8, y_solid(1.8)+0.01, "SINDy", color="b", verticalalignment="bottom")
plt.text(1.8, y_solid(1.8)+0.4, "Parsimonious\napproximation", color="lightblue", verticalalignment="top", horizontalalignment="center")

plt.plot(5, y_solid(5), 'b.', zorder=3)
plt.text(5, y_solid(5), "(Least square)\n$\lambda = 0$", color="b", verticalalignment="bottom",horizontalalignment="center")

plt.plot(5, 0.38, 'b.', zorder=3)
plt.text(5, 0.42, "(Overfit)", color="b", verticalalignment="bottom",horizontalalignment="center")

# Remove axes and frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.box(on=None)  # Or plt.gca().set_frame_on(False)

# Draw the X and Y axes
plt.axhline(y=-0.1, color='black', linewidth=0.5)  # Horizontal line for X-axis
plt.axvline(x=-0.2, color='black', linewidth=0.5)  # Vertical line for Y-axis

plt.show()

# Save the fig.
# save_image(fig, "pareto_curve")

# Fin :)