# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Fashion MNIST data from CSV
file_path = '/content/fashion-mnist_train.csv'
data = pd.read_csv(file_path)

# Handle missing values by filling with column means
data.fillna(data.mean(), inplace=True)

# Separate features (pixel values) and labels
labels = data['label']
features = data.drop('label', axis=1)

# Standardize the features (recommended for PCA)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA to reduce the dimensionality to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# Add the PCA results to a new DataFrame for visualization
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['label'] = labels

# Plot the 2D PCA visualization for all classes
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df, palette='Set1', alpha=0.6)
plt.title('2D PCA of Fashion MNIST dataset')
plt.show()
