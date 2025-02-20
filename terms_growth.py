"""
Growth in the number of terms in a polynomial feature library in the sparse 
identification of nonlinear dynamics (SINDy) algorithm.

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia
    
Date:
    November 2024
"""


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.special import gamma

from functions import save_image

def calculate_polynomial_vector_size(n: int, p: int) -> int:
    """
    Calculate the size 'q' of the vector of polynomial terms up to degree 'p',
    given a vector with 'n' initial terms.

    Args:
        n (int): Number of terms in the initial vector.
        p (int): Maximum degree of polynomial terms.

    Returns:
        int: Size 'q' of the resulting polynomial terms vector.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if p < 1:
        raise ValueError("p must be at least 1")

    q = sum(math.comb(n + k - 1, k) for k in range(p + 1))
    
    return q

# Example usage and testing.
if __name__ == "__main__":
    
    # -----
    # Test the function performance.
    test_cases = [
        (5, 1),  # Linear terms only
        (5, 2),  # Up to quadratic terms
        (5, 3),  # Up to cubic terms
        (3, 4),  # Different n and p
        (10, 5)  # Larger n and p
    ]

    for n, p in test_cases:
        q = calculate_polynomial_vector_size(n, p)
        print(f"For n = {n} initial terms and up to degree p = {p}:")
        print(f"The size of the polynomial terms vector is q = {q}")
        print()
    
    # -----
    # Analyse the growth of the polynomial feature library with a initial 
    # vector size.
    
    # The initial vector size, polynomial terms of degree = 1
    y = ['x', 'x_dot', 'a_h', '|x_dot|', '|a_h|']
    n = len(y)
    nn = np.arange(0, n)

    # We want polynomials from p_min up to p_max-degree
    p_min = 1
    p_max = 10

    # Vector with the degrees of polynomial order up to which we want to know the
    # number of terms.
    pp = np.arange(p_min, p_max)
    
    # Vector with the number of terms.
    qq = np.zeros(p_max-p_min, dtype=int)
    
    for degree in pp:
        index = degree - p_min
        qq[index] += int(calculate_polynomial_vector_size(n, degree))
        print(f"p = {degree} ---> q = {qq[index]}")
    
    # -----
    # Analyse the exponential and combinatorial growth.
    
    # Define the exponential function
    exponential_func = lambda x, a, b : a * np.exp(b * x)
    # Fit the curve witha an exponential growth
    params_exp, _ = curve_fit(exponential_func, pp, qq)
    a, b = params_exp
    # Create the fitted curve exponential
    y_fit_exp = exponential_func(pp, a, b)
    
    # Plot the data and the fitted curve
    fig = plt.figure(figsize=(7, 3))
        
    plt.scatter(pp, qq, label='# of terms', c="k")
    plt.plot(pp, y_fit_exp, "b-", label='Exponential fit')
    plt.xlabel("polynomial order ($p$)") # pp
    plt.ylabel("# of terms ($q$) up to a given\npolynomial order ($p$)") # qq
    plt.title(f"For $n$ = {n}, $p$ up to {p_max-1}") 
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    
    # Store the figure.
    save_image(fig, "terms-growth")