"""
Some functions to optimize the SINDy implementation for identification of 
hysteresis.

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

References:
    Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing 
    equations from data by sparse identification of nonlinear dynamical systems.
    Proceedings of the national academy of sciences, 113(15), 3932-3937.
    
    de Silva, B. M., Champion, K., Quade, M., Loiseau, J. C., Kutz, J. N., & 
    Brunton, S. L. (2020). Pysindy: a python package for the sparse 
    identification of nonlinear dynamics from data. arXiv preprint 
    arXiv:2004.08424.

    Kaptanoglu, A. A., de Silva, B. M., Fasel, U., Kaheman, K., Goldschmidt, 
    A. J., Callaham, J. L., ... & Brunton, S. L. (2021). PySINDy: A 
    comprehensive Python package for robust sparse system identification. arXiv 
    preprint arXiv:2111.08481.
 
    Oficial documentation at: 
    https://pysindy.readthedocs.io/en/latest/index.html

    Oficial GitHub repository at: 
    https://github.com/dynamicslab/pysindy/tree/master

Date:
    January 2025
"""


import numpy as np
import pysindy as ps

from functions import plot_comparatives_3, save_image
from abs_smooth_approximation import abs_smooth_approximation


def simulate_sindy_model(
    X, X_0, t, ft_lbs, th, control_u=False, store=False
    ) -> None:
    """
    This functions generates a SINDy model from the data and performes the 
    simulation based on given initial conditions. The feature library 
    information for the modeling process is given by a dictionary entry. 
    The algorithm is the STLSQ with a fixed alpha, and the method for 
    numerical differentiation is the default one. 
    
    Args:
        X (array): 
            State vector.
        X_0 (array): 
            Initial conditions.
        t (array): 
            Time vector.
        ft_lbs (dic): 
            - Its keys (string) are the coded names of the feature library.
            - Its values (pysindy objects) are the ps.FeatureLibrary structure 
            with its own constraints.
        th (float): Threshold for the optimizer, the STLSQ.
        optimizer (str): default to "sltsq".
            The choosen optimizer to peform SINDy (stlsq, lasso).
        control_u (numpy.ndarray): default to False.
            Control vector: external excitation or contorl input. 
        store (str): defaul to False.
            Wether or no to store the plot. To store it, the directory folder 
            name must be given.
    
    """
    for ft_lb_key, ft_lb_val in ft_lbs.items():
        
        # Define the optimization method for the model.
        optimizer = ps.STLSQ(threshold=th, alpha=0.05)
        
        # Build the SINDy model with the defined feature library.
        model = ps.SINDy(feature_library=ft_lb_val, optimizer=optimizer)
        
        # Fit the model, compute the simulation for the initial conditions and 
        # get a score.
        if control_u is not False:  # This ensures only when a valid array is provided
            model.fit(X, t=t, u=control_u)
            X_sindy = model.simulate(X_0, t, u=control_u)
            score = model.score(X, t=t, u=control_u)
            # FIX THE SIZE PROBLEM: actually do not know why this fails.
            X_sindy = np.vstack([X_sindy, X_sindy[-1,:]])
        else:
            model.fit(X, t=t)
            X_sindy = model.simulate(X_0, t) 
            score = model.score(X, t)

            
        # Repor the model, its score and the features implemented.
        print("\n-----\n")
        print(f"Used library: {ft_lb_key}")
        print("\nThe model:"); model.print()
        print("\nThe score:"); print(score)
        print("\nThe names:"); print(model.get_feature_names())

        # Plot the results.
        fig = plot_comparatives_3(
            X.T, X_sindy.T, t, xylabels=["x(t)", "fr(t)", "t"], title=ft_lb_key
            )
        
        if store is not False:
            save_image(fig, ft_lb_key, directory=store)


def build_pos_polynomial_libraries(up_to_degree=3):
    """
    This function build possible polynomial libraries to try the SINDY 
    identification process, taking into consideration polynoials up to 3rd 
    degree and wether bias and terms interaction is accoutnable.  

    Returns:
        (dict): 
            dictionary with the possible polynomial feature libraries. 
            - Its keys (string) are the coded names of the feature library.
            - Its values (pysindy objects) are the ps.FeatureLibrary structure 
            with its own constraints.
    """
    # Argument options for Polynomial Library
    pos_pol_degress = range(1,up_to_degree+1)   # Possible degrees
    pos_bias =  [True, False]                   # Possible bias 
    pos_interaction = [True, False]             # Possible terms interaction

    # Define the possible polynomial feature libraries
    poly_feature_libraries = {}

    # Store the possible combinations of polynomial libraries.
    for i in pos_interaction:
        for j in pos_bias:
            for n in pos_pol_degress:
                poly_feature_libraries[f"FL_pol_n{n}_b{j}_i{i}"] = ps.PolynomialLibrary(
                    degree=n, 
                    include_interaction=i, 
                    include_bias=j
                    )
    
    return poly_feature_libraries

def build_pos_fourier_libraries(up_to_freq=4):
    """
    This function build possible Fourier libraries to try the SINDY 
    identification process, taking into consideration up to 4 frequencies.

    Returns:
        (dict): 
            dictionary with the possible Fourier feature libraries. 
            - Its keys (string) are the coded names of the feature library.
            - Its values (pysindy objects) are the ps.FeatureLibrary structure 
            with its own constraints.
    """
    # Arguments options for the Fourier Library
    pos_freq = range(1,up_to_freq+1)                   # Possible frequencies
    # Define the possible polynomial feature libraries
    four_feature_libraries = {}

    # Considers both sine and cosine terms.
    for freq in pos_freq:
        four_feature_libraries[f"FL_fr_n{freq}"] = ps.FourierLibrary(
            n_frequencies=freq
            )
    
    return four_feature_libraries

def build_pos_custom_library():
    """
    This function builds custom lñibrary to try the SINDY 
    identification process, taking into consideration the functions: absolute 
    value (approx), sign (approx), exponential, sin(A+B), cos(A+B).  

    Returns:
        (dict): 
            dictionary with the custom feature library. 
            - Its key (string) is the coded name of the feature library.
            - Its value (pysindy object) is the ps.FeatureLibrary structure.
    """
    # Functions for Custom Library
    pos_custom_functions = [                   # Names, if not given above. 
        lambda x: abs_smooth_approximation(x), # f1(.) --> Absolute value.
        lambda x: np.tanh(10*x),               # f2(.) --> Sign function approx.
        lambda x: np.exp(x),                   # f3(.) --> Exponential function. 
        #lambda x: 1/x,                         # f4(.) --> 1/x  
        lambda x, y: np.sin(x + y),            # f5(.) --> Sin of A + B 
        lambda x, y: np.cos(x + y),            # f6(.) --> Cos of A + B 
    ]

    # Names for the functions.
    pos_custom_fun_names = [
        lambda x: "abs(" + x + ")",
        lambda x: "sgn(" + x + ")",
        lambda x: "exp(" + x + ")",
        #lambda x: "1/" + x,
        lambda x, y: "sin(" + x + "," + y + ")",
        lambda x, y: "cos(" + x + "," + y + ")"
    ]
    
    # Define the possible custom functions without a given name.
    cust_feature_library = {} 
    cust_feature_library["FL_cus"] = ps.CustomLibrary(
        library_functions=pos_custom_functions,
        function_names=pos_custom_fun_names,
        interaction_only=False # to consider all types of cominations
        )
    
    return cust_feature_library

 
# Fin :)