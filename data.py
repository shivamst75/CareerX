import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting library

# Load the dataset from CSV
data = pd.read_csv('C:\\Users\\Admin\\OneDrive\\Desktop\\careerx model\\CareeX Data FINAL NEW.csv')

# Assuming 'Total_Score', 'IQ_Score', and 'EQ_Score' are the features you want to use for classification
X = data[['TOTAL', 'IQ', '10TH']]
y = data['STREAM']  # Adjust the column name according to your dataset

# SVM classifier
clf = SVC(kernel='linear')

# Fit the classifier to the data
clf.fit(X, y)

# Define a meshgrid for the 3D plot
total_score_range = np.linspace(X['TOTAL'].min(), X['TOTAL'].max(), 100)
iq_score_range = np.linspace(X['IQ'].min(), X['IQ'].max(), 100)
eq_score_range = np.linspace(X['10TH'].min(), X['10TH'].max(), 100)
total_score_mesh, iq_score_mesh, eq_score_mesh = np.meshgrid(total_score_range, iq_score_range, eq_score_range)

# Predict the classification for each point in the meshgrid
Z = clf.predict(np.c_[total_score_mesh.ravel(), iq_score_mesh.ravel(), eq_score_mesh.ravel()])
Z = Z.reshape(total_score_mesh.shape)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the decision boundary surface using the viridis colormap
#ax.plot_surface(total_score_mesh, iq_score_mesh, eq_score_mesh, cmap='viridis', alpha=0.5)
#ax.plot_surface(total_score_mesh, iq_score_mesh, eq_score_mesh, facecolors=plt.cm.viridis(Z), alpha=0.5)
ax.plot_trisurf(total_score_mesh.ravel(), iq_score_mesh.ravel(), eq_score_mesh.ravel(), triangles=None, cmap='viridis', alpha=0.5)

# Plot data points
for label in np.unique(y):
    ax.scatter(X.loc[y == label, 'TOTAL'], X.loc[y == label, 'IQ'], X.loc[y == label, '10TH'], label=label)

ax.set_xlabel('Total Score')
ax.set_ylabel('IQ Score')
ax.set_zlabel('EQ Score')
ax.set_title('Classification in 3D Space with SVM')
ax.legend()
plt.show()
