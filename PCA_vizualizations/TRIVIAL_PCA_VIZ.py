import numpy as np
import pickle as pkl
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load trivial TEST2 data from course
data_path = "/content/index.pkl"
with open(data_path, 'rb') as file:
    index_dicts = pkl.load(file)

# Data initialization
index_embeddings = [d["embeddings"] for d in index_dicts]
TRIVIAL_embeddings_array = np.array(index_embeddings)
TRIVIAL_embedding_dim = TRIVIAL_embeddings_array.shape[1]
index_categories = [d["category"] for d in index_dicts]
mapping = {"Beauty": 0, "Software": 1, "Appliances": 2}
TRIVIAL_labels = [mapping[item] for item in index_categories]

# Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
TRIVIAL_embeddings_2d = pca.fit_transform(TRIVIAL_embeddings_array)

# Plot the 2D PCA visualization
plt.figure(figsize=(10, 7))
for label in np.unique(TRIVIAL_labels):
    plt.scatter(
        TRIVIAL_embeddings_2d[np.array(TRIVIAL_labels) == label, 0], 
        TRIVIAL_embeddings_2d[np.array(TRIVIAL_labels) == label, 1], 
        label=list(mapping.keys())[list(mapping.values()).index(label)]
    )
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("2D PCA Visualization of Embeddings")
plt.legend()
plt.show()
