"""
Matplotlib rcParams configuration for academic publications.

This module provides standardized matplotlib configuration settings optimized for 
Springer Nature journal publications, using STIX fonts for consistency with LaTeX 
documents and ensuring proper EPS compatibility.

Coder:
   Michael Heredia Pérez
   mherediap@unal.edu.co
   Universidad Nacional de Colombia 
   170001 Manizales, Colombia
   
Date:
   May 2025
"""


import matplotlib as mpl

def configure_matplotlib():
    """Configure matplotlib for Springer Nature publication standards with STIX font"""
    mpl.rcParams.update({
        # Primary font configuration
        "font.family": "serif",
        "font.serif": ["STIX", "STIXGeneral", "STIX Two Text"],
        "mathtext.fontset": "stix",
        
        # Font sizes for journal publication
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        
        # Line properties
        "lines.linewidth": 1.0
    })

# Auto-configure when imported
configure_matplotlib()