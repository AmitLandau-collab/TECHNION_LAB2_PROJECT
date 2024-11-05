import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
file_path = '/content/fashion-mnist_train.csv'
df = pd.read_csv(file_path)
pixel_columns = [col for col in df.columns if 'pixel' in col]
MMNIST_labels = df['label'].values
MMNIST_embeddings_array = df[pixel_columns].values  # Get the pixel columns as a NumPy array

# Filter the dataset for classes 2 and 4
mask = np.isin(MMNIST_labels, [2, 4])
filtered_labels = MMNIST_labels[mask]
filtered_embeddings = MMNIST_embeddings_array[mask]

# Separate class 2 and class 4
class_2_vectors = filtered_embeddings[filtered_labels == 2]
class_4_vectors = filtered_embeddings[filtered_labels == 4]

# Add Gaussian noise to class 2 vectors to create a larger dataset
def add_gaussian_noise(data, noise_level=0.1):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# Create 3 times more class 2 vectors
num_class_2_to_generate = len(class_2_vectors) * 2  # we want to triple the count, so we need to generate 2 additional for each
new_class_2_vectors = np.array([add_gaussian_noise(vec) for vec in class_2_vectors for _ in range(3)])[:num_class_2_to_generate]

# Combine the new dataset
final_embeddings = np.vstack((new_class_2_vectors, class_2_vectors, class_4_vectors))
final_labels = np.array([2] * len(new_class_2_vectors) + [2] * len(class_2_vectors) + [4] * len(class_4_vectors))
# PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(final_embeddings)

# Plotting the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[final_labels == 2][:, 0], reduced_embeddings[final_labels == 2][:, 1],
            label='Class 2', alpha=0.8, color='blue')
plt.scatter(reduced_embeddings[final_labels == 4][:, 0], reduced_embeddings[final_labels == 4][:, 1],
            label='Class 4', alpha=0.3, color='orange')
plt.title('PCA of MNIST Classes 2 and 4 with Augmented Class 2')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid()
plt.show()
