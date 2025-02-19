"""
Approximation of the sign function with the help of the hyperbolic tangent 
function.

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia
    
Date:
    Januray 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from abs_smooth_approximation import abs_smooth_approximation
from functions import save_image

# -----
# Plot the sign approximation. 

x_1     = np.linspace(-10, 10, 500)
sign_np = np.sign(x_1)

fig_sign = plt.figure()

plt.plot(x_1, np.tanh(x_1),    "b-",  label="y=tanh(x)")
plt.plot(x_1, np.tanh(10*x_1), "r-",  label="y=tanh(10x)")
plt.plot(x_1, sign_np,         "g--", label="y=sign(x)")
plt.grid(True, alpha=0.5)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--") 
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.show()

# save_image(fig_sign, "approx-sign-tanh")


# -----
# Plot the abs approximation. 

x_2 = np.linspace(-0.5, 0.5, 500)

# Setting the abs value approximation.
abs_np         = np.abs(x_2)
abs_smooth_10  = abs_smooth_approximation(x_2, k=10)
abs_smooth_100 = abs_smooth_approximation(x_2, k=100)

fig_abs = plt.figure()

plt.plot(x_2, abs_np,     "b-",  label="y=abs(x)")
plt.plot(x_2, abs_smooth_10, "g-",  label="smooth abs, k=10")
plt.plot(x_2, abs_smooth_100, "r-",  label="smooth abs, k=100")
plt.grid(True, alpha=0.5)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--") 
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.show()

# save_image(fig_abs, "approx-abs-smooth")

# Fin :)