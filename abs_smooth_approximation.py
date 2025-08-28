"""
This is a comparison of the absolute value function calculated by Python vs the 
smooth approximation taught by Diego.

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia
    
Date:
    Januray 2025
"""

import matplotlib.pyplot as plt
import random
import numpy as np

# Configure matplotlib for STIX font - comprehensive setup
import matplotlib as mpl
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

def abs_smooth_approximation(x, k=10):
    """This function calculates the smoth approximation of the absolute value of
    a given number by means of the formula:

    |x| \approx 2*x / pi atan (k*x) 

    Args:
    -----
        x (float): desired number to calculate its absolute value.
        k (int): Default = 10. Parameter > 0 for the approximation. 
    
    Returns:
    --------
        float: absolute value approximation.
    """
    return 2*x/np.pi*np.arctan(k*x)


# Execute comparison analysis.
if __name__ == "__main__":

    # -----
    # Evaluate the approximation.
    # This example was coded withoyt the use of numpy just because. 
    # A beter implementation could be made.

    # Generate random integers.
    lims = 100
    nnum = 6
    random_numbers = [random.randint(-lims, lims) for _ in range(nnum)]

    # Calculate the absolute value of the random numbers.
    abs_python = []
    abs_approx = []
    for number in random_numbers:
        abs_python.append(abs(number))
        abs_approx.append(abs_smooth_approximation(number))

    # -----
    # Present the results in an aligned table format

    print("\n--- Abs values for k = 10.")
    print(f"{'Random numbers:':<30} {random_numbers}")
    print(f"{'Absolute value Python:':<30} {abs_python}")
    print(f"{'Absolute value approximation:':<30} {abs_approx}")

    # -----
    # Verify the variation of the parameter k.

    # Create a list of possible values of k.
    kk = [k for k in range(1,15)]

    # Empties lists to store the calculation and the respective errors. 
    abs_approx_kk = []
    absolute_error = []

    # Calculate each random number's abs value varying the parameter k.
    for i, number in enumerate(random_numbers):

        # a meantime empty list. 
        abs_approx_meantime = []
        error_meantime = []

        # Vary k.
        for k in kk:
            abs_approx_meantime.append(abs_smooth_approximation(number, k=k))
            # Calculate and store the absolute error.
            error = (abs_python[i] - abs_smooth_approximation(number, k=k)) / abs_python[i]
            # round the absolute error.
            error_meantime.append(round(error, 2))

        # Store the abs calculated for each k and its absolute erro.
        abs_approx_kk.append(abs_approx_meantime)
        absolute_error.append(error_meantime)

    # -----
    # Plot the results

    # Create subplots
    fig, axes = plt.subplots(3, 2, sharex=True)

    # Plot for each random number
    for i, ax in enumerate(axes.flatten()):
        
        # Plot the absolute value by python and by approximation 
        ax.plot(kk, [abs_python[i]]*len(kk), "k*--", label="Python abs(.)")
        ax.plot(kk, abs_approx_kk[i], "r*--", label=r"$|x| \approx (2x / pi) \arctan (kx)$")

        # Plot the absolute error at each k.
        for j, k in enumerate(kk):
            if j == 0:
                ax.text(k, abs_approx_kk[i][j], f"$\epsilon=${absolute_error[i][j]}")
            else:
                ax.text(k, abs_approx_kk[i][j], absolute_error[i][j])


        ax.grid()
        # Add labels and legend only on the first plot
        if i == 0:
            ax.set_xlabel("Parameter k")
            ax.set_ylabel("Absolute Error")
            ax.legend(loc="lower right")
        else:
            # Remove labels for other subplots
            ax.set_xlabel("")
            ax.set_ylabel("")

        # Title for each subplot
        ax.set_title(f"Random Number: {random_numbers[i]}")

    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()

# Fin :)