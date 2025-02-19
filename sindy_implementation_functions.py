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
import matplotlib.pyplot as plt

import pysindy as ps
from sklearn.linear_model import Lasso

from functions import plot_comparatives_3, save_image


def simulate_sindy_model(
    X, X_0, t, ft_lbs, optimizer="stlsq", control_u=False, store=False
    ) -> None:
    """
    This functions generates a SINDy model from the data and performes the 
    simulation based on given initial conditions. The feature library 
    information for the modeling process is given by a dictionary entry. 
    The algorithm is the STLSQ with a fixed values, and the method for 
    numerical differentiation is the default one. 
    
    Args:
        X (array): 
            State vector.
        X_0 (array): 
            Initial conditions.
        t (array): 
            Time vector.
        ft_lbs (dic): 
            Its keys (string) are the coded names of the feature library.
            - Its values (pysindy objects) are the ps.FeatureLibrary structure 
            - with its own constraints.
        optimizer (str): default to "sltsq".
            The choosen optimizer to peform SINDy (stlsq, lasso).
        control_u (numpy.ndarray): default to False.
            Control vector: external excitation or contorl input. 
        store (boolean): defaul to False.
            Wether or no to store the plot.
    
    """
    for ft_lb_key, ft_lb_val in ft_lbs.items():
        
        # Build the SINDy model with the defined feature library.
        model = ps.SINDy(feature_library=ft_lb_val)

        # Define the optimization method for the model.
        if optimizer=="stlsq":
            model.optimizer = ps.STLSQ(threshold=0.01, alpha=0.05)
        elif optimizer=="lasso":
            model.optimizer = Lasso(alpha=2, max_iter=2000, fit_intercept=False)
        else:
            raise Exception("The optimizer chosen is not valid in this code.")
        
        # Fit the model, compute the simulation for the initial conditions and 
        # get a score.
        if control_u is not False:  # This ensures only when a valid array is provided
            model.fit(X, t, u=control_u)
            X_sindy = model.simulate(X_0, t, u=control_u) 
            score   = model.score(X, t, u=control_u)
            # FIX THE SIZE PROBLEM: actually do not know why this fails.
            X_sindy = np.vstack([X_sindy, X_sindy[-1,:]])
        else:
            model.fit(X, t=t)
            X_sindy = model.simulate(X_0, t) 
            score   = model.score(X, t)

            
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
        
        if store:
            save_image(fig, ft_lb_key, directory="sindy_plots_2")


def build_polynomial_libraries():
    pass

def build_fourier_libraries():
    pass

def build_custom_libraries():
    pass

 


# Fin :)
