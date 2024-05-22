import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data from CSV file
data = pd.read_csv('C:\\Users\\Admin\\OneDrive\\Desktop\\careerx model\\CAREERX RESULTS.csv')

# Assuming your CSV file has columns named Parameter1, Parameter2, ..., Parameter6
parameters = ['IQ ', 'O', 'C', 'E', 'A', 'N']

# Define the Gaussian function
def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Perform Gaussian curve fitting for each parameter
fits = {}
for param in parameters:
    # Fit the Gaussian curve
    hist, bin_edges = np.histogram(data[param], bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[np.mean(data[param]), np.std(data[param]), 1], maxfev=1000000)

    # Store the fitting parameters
    fits[param] = popt

# Plot the results
plt.figure(figsize=(12, 8))
for i, param in enumerate(parameters):
    plt.subplot(2, 3, i+1)
    plt.hist(data[param], bins=30, density=True, alpha=0.5, label='Data')
    plt.plot(bin_centers, gaussian(bin_centers, *fits[param]), color='red', label='Gaussian Fit')
    plt.title(param)
    plt.legend()
plt.tight_layout()
plt.show()