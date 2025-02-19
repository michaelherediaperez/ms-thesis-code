"""
Evolution of the Bouc-Wen class citations over the years.

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

Date:
    Januray 2025
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from functions import save_image

# fontsize
ft = 14

# Define file paths
file1 = "scopus/bw_yearly_counts_specific.txt"   
file2 = "scopus/bw_yearly_counts_general.txt"    

# Read both files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Set the plot style
sns.set_style("darkgrid")

# Create plots for each dataset
for i, df in enumerate([df1, df2], start=1):
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(x="Year", y="Count", data=df, marker="o", linewidth=2.5, color='b')
    plt.xlabel("Year", fontsize=ft, labelpad=10)
    plt.ylabel("Number of Publications", fontsize=ft, labelpad=10)
    plt.xticks(rotation=45, fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()
    
    # Save the figure with a descriptive filename
    filename = f"publications_over_time_bw_dataset_{i}"
    # save_image(fig, filename, directory="scopus_plots")
    plt.close()  # Close the figure to free up memory