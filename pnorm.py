"""
Graphs for the ilustration of the p-norms (individually): L0, L1, and L2.
This is not a matheamtical example, just a mock-up.

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia

January 2025
"""


import numpy as np
import matplotlib.pyplot as plt
from functions import save_image

import matplotlib as mpl
# Configure matplotlib for STIX font - comprehensive setup
mpl.rcParams.update({
    # Primary font configuration
    "font.family": "serif",              # Use serif family
    "font.serif": ["STIX", "STIXGeneral", "STIX Two Text"], # STIX font priority
    "mathtext.fontset": "stix",          # Math expressions in STIX
    
    # Explicit font specification for all text elements
    "axes.labelsize": 18,
    "axes.titlesize": 18, 
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.size": 16,
    
    # Line properties
    "lines.linewidth": 1.5
})

#plot_kws = dict(linewidth=1, alpha=0.7)

# Parameters for the orange line (representing the solution space)
def solution_space(x):
    return -0.5 * x + 1.5  

# The lims for the axes
ax_lim = 2.5

x_vals = np.linspace(-ax_lim, ax_lim, 2)
y_vals = solution_space(x_vals)


def plot_pnorm(norm, radious, sol, inter, title, ax_lim=ax_lim):
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.set(xlim=(-ax_lim, ax_lim), ylim=(-ax_lim, ax_lim), xlabel=f"$s_1$", ylabel=f"$s_2$")
    # Plot the solution line.
    ax.plot(x_vals, y_vals, color="red", label="Family of solutions")

    if norm=="l2":
        for r in radious:
            circle = plt.Circle((0, 0), r, color="blue", fill=False)
            ax.add_artist(circle)
                
    elif norm=="l0":
        new_lim = ax_lim - 1
        ax.plot([-new_lim, new_lim], [0, 0], color="blue")
        ax.plot([0, 0], [-new_lim, new_lim], color="blue")

    elif norm == "l1":
        for r in radious:
            diamond = np.array([[r, 0], [0, r], [-r, 0], [0, -r], [r, 0]])
            ax.plot(diamond[:, 0], diamond[:, 1], color="blue")
        
    else:
        raise Exception(f"{norm} is not valid.")
    
    ax.scatter([sol], [solution_space(sol)], color="green", label=f"Solution ({sol}, {inter})", zorder=5)
    #ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


# -----
# Plot for L2-norm 
# Intersects the solution with the minimum radious 

# The intersection point (for the given line equation).
x_l2 = 3/5
r_l2 = 1.34

#radious_l2 = [r_l2-1.25, r_l2-1.0, r_l2-0.75, r_l2-0.5, r_l2-0.25, r_l2]
radious_l2 = [r_l2-0.34*3, r_l2-0.34*2, r_l2-0.34, r_l2]

fig_l2 = plot_pnorm(norm="l2", radious=radious_l2, sol=x_l2, inter=r_l2, title="L2-Norm constraint")
save_image(fig_l2, "l2-norm")

# -----
# Plot for L0-norm 

# The intersection point.
x_l0 = 0
r_l0 = 1.5

fig_l0 = plot_pnorm(norm="l0", radious=radious_l2, sol=x_l0, inter=r_l0, title="L0-Norm constraint")
save_image(fig_l0, "l0-norm")

# -----
# Plot for L1-norm 

radious_l1 = [r_l0-0.34*3, r_l0-0.34*2, r_l0-0.34, r_l0]

fig_l1 = plot_pnorm(norm="l1", radious=radious_l1, sol=x_l0, inter=r_l0, title="L1-Norm constraint")
save_image(fig_l1, "l1-norm")
