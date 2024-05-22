import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('C:\\Users\\Admin\\OneDrive\\Desktop\\careerx model\\CAREERX RESULTS.csv')

# Assuming your CSV file has columns named Parameter1, Parameter2, ..., Parameter6
parameters = ['IQ ', 'O', 'C', 'E', 'A', 'N']

# Create violin plots
plt.figure(figsize=(12, 8))
for i, param in enumerate(parameters):
    plt.subplot(2, 3, i+1)
    sns.violinplot(data[param])
    plt.title(param)
plt.tight_layout()
plt.show()
