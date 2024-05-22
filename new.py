import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset from CSV
data = pd.read_csv('C:\\Users\\Admin\\OneDrive\\Desktop\\careerx model\\CareeX Data FINAL NEW.csv')

# Assuming 'IQ', '10TH', 'STREAM VALUE', and 'STREAM' are the features you want to use for classification
X = data[['IQ', '10TH', 'STREAM VALUE']]
y = data['STREAM']  
# SVM classifier
clf = SVC(kernel='rbf', gamma='scale')  # Using radial basis function kernel

# Fit the classifier to the data
clf.fit(X[['IQ', '10TH']], y)  # Considering only IQ and 10th percentage for prediction

# Function to predict stream based on user input
def predict_stream(iq, tenth):
    # Predict the stream using SVM classifier
    predicted_stream = clf.predict([[iq, tenth]])[0]
    stream_value = data[data['STREAM'] == predicted_stream]['STREAM VALUE'].iloc[0]
    return predicted_stream, stream_value

# User input for new data
user_iq = float(input("Enter IQ score: "))
user_tenth = float(input("Enter 10th percentage: "))

# Predict stream for new data
predicted_stream, predicted_stream_value = predict_stream(user_iq, user_tenth)
print("Predicted stream:", predicted_stream)

# Add user input as a new data point
new_data_point = pd.DataFrame({'IQ': [user_iq], '10TH': [user_tenth], 'STREAM': [predicted_stream], 'STREAM VALUE': [predicted_stream_value]})
data = pd.concat([data, new_data_point], ignore_index=True)

# Calculate accuracy
y_pred = clf.predict(X[['IQ', '10TH']])
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Plot decision boundary
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define range for IQ and 10TH
IQ_range = np.linspace(X['IQ'].min(), X['IQ'].max(), 50)
tenth_range = np.linspace(X['10TH'].min(), X['10TH'].max(), 50)

# Create meshgrid for IQ and 10TH
IQ_mesh, tenth_mesh = np.meshgrid(IQ_range, tenth_range)

# Predict the class for each point in the meshgrid
Z = clf.predict(np.c_[IQ_mesh.ravel(), tenth_mesh.ravel()])
Z = Z.reshape(IQ_mesh.shape)

# Map stream to color
color_map = {'Arts': 'red', 'Commerce': 'blue', 'PCM': 'green', 'PCB': 'orange', 'PCB ': 'yellow'}

# Plot decision boundary
for stream, color in color_map.items():
    ax.scatter(X['IQ'][y == stream], X['10TH'][y == stream], X['STREAM VALUE'][y == stream], color=color, label=stream)

# Plot user input as a new data point
if predicted_stream in color_map:
    ax.scatter(user_iq, user_tenth, predicted_stream_value, color='black', marker='.', s=500, label='User Input')
else:
    print("Warning: Predicted stream value not found in color map. Plotting as default color.")
    ax.scatter(user_iq, user_tenth, predicted_stream_value, color='black', marker='x', s=100, label='User Input')

ax.set_xlabel('IQ')
ax.set_ylabel('10TH')
ax.set_zlabel('STREAM VALUE')
ax.set_title('SVM Decision Boundary in 3D')
plt.legend()
plt.show()
