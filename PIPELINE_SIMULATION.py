import numpy as np
import faiss
import random
from collections import Counter
from scipy.stats import entropy
import pickle as pkl
import itertools
import pandas as pd
from sklearn.metrics import f1_score
import copy

# Add Gaussian noise to class 2 vectors to create a larger dataset
def add_gaussian_noise(data, noise_level=0.1):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# Data Split
def split_data(labels, embeddings, init_data_size=800, test_size=4400):
    total_size = len(labels)
    indices = list(range(total_size))
    random.shuffle(indices)

    test_indices = indices[:test_size]
    init_data_indices = indices[test_size:test_size + init_data_size]
    sample_pool_indices = indices[test_size + init_data_size:]

    test_labels = [labels[i] for i in test_indices]
    test_embeddings = [embeddings[i] for i in test_indices]
    init_data_labels = [labels[i] for i in init_data_indices]
    init_data_embeddings = [embeddings[i] for i in init_data_indices]
    sample_pool_labels = [labels[i] for i in sample_pool_indices]
    sample_pool_embeddings = [embeddings[i] for i in sample_pool_indices]

    return (init_data_labels, init_data_embeddings, sample_pool_labels,
            sample_pool_embeddings, test_labels, test_embeddings)

# FAISS Index Initialization Methods
def init_L2(vector_embeddings, d):
    index = faiss.IndexFlatL2(d)
    index.add(vector_embeddings)
    return index

def init_IVF(vector_embeddings, d, nlist=50):
    middle_step = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(middle_step, d, nlist)
    index.train(vector_embeddings)
    index.add(vector_embeddings)
    return index

def init_LSH(vector_embeddings, d, nbits=None):
    if not nbits:
        nbits = d * 2
    index = faiss.IndexLSH(d, nbits)
    index.add(vector_embeddings)
    return index

def init_HNSW(vector_embeddings, d, M=64, ef_search=32, ef_construction=64):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vector_embeddings)
    return index


def compute_f1_score(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, average='weighted')

# Optimized most_common_label with NumPy
def most_common_label(labels, values_to_check):
    count = Counter(labels)
    total = len(labels)
    return np.array([count[value] / total if value in count else 0 for value in values_to_check])

# Optimized get_pred_vectors with batch search and NumPy processing
def get_pred_vectors(index, sample_pool, init_data_labels, k, labels_range):
    # Run batch search for all samples in sample_pool at once
    D, I = index.search(np.array(sample_pool), k)  # I is of shape (len(sample_pool), k)

    # Convert init_data_labels to a NumPy array for faster indexing
    init_data_labels = np.array(init_data_labels)

    # Use list comprehension and NumPy vectorized operations to obtain predictions
    preds = [
        most_common_label(init_data_labels[sample_indices], labels_range)
        for sample_indices in I
    ]

    return np.array(preds)

# Active Learning
def select_samples(preds, samples_per_iter=100, entropy_weight=0.9, least_confident_weight=0.0, margin_weight=0.1):
    preds = np.array(preds)

    if not np.isclose(entropy_weight + least_confident_weight + margin_weight, 1.0):
        raise ValueError("The sum of weights must be equal to 1.")

    entropies = entropy(preds.T)
    most_likely_probabilities = np.max(preds, axis=1)
    least_confident_scores = 1 - most_likely_probabilities
    sorted_probabilities = np.sort(preds, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    entropy_max = np.max(entropies)
    least_confident_max = np.max(least_confident_scores)
    margin_max = np.max(margins)

    normalized_entropy = entropies / entropy_max if entropy_max > 0 else entropies
    normalized_least_confident = least_confident_scores / least_confident_max if least_confident_max > 0 else least_confident_scores
    normalized_margin = margins / margin_max if margin_max > 0 else margins

    combined_scores = (entropy_weight * normalized_entropy +
                       least_confident_weight * normalized_least_confident +
                       margin_weight * normalized_margin)

    return np.argsort(combined_scores)[-samples_per_iter:]

# Random Sample Selection instead of AL
def select_random_samples(sample_pool_size, samples_per_iter=100):
    return random.sample(range(sample_pool_size), samples_per_iter)

# Transfer Samples to ANN Index
def transfer_samples(sample_pool_embeddings, sample_pool_labels, init_data_embeddings, init_data_labels, indexes, index):
    elements_to_transfer = [sample_pool_embeddings[i] for i in indexes]
    labels_to_transfer = [sample_pool_labels[i] for i in indexes]

    init_data_embeddings = np.append(init_data_embeddings, np.array(elements_to_transfer), axis=0)
    init_data_labels.extend(labels_to_transfer)

    for i in sorted(indexes, reverse=True):
        del sample_pool_embeddings[i]
        del sample_pool_labels[i]

    index.add(np.array(elements_to_transfer))
    return sample_pool_embeddings, sample_pool_labels, init_data_embeddings, init_data_labels, index

def evaluate_F1(index, test_embeddings, test_labels, init_data_labels, k, labels_range): ### macro f1
    test_preds = get_pred_vectors(index, test_embeddings, init_data_labels, k, labels_range)
    # Use argmax to find the index of the maximum value within the NumPy array
    predicted_labels = [np.argmax(lst) for lst in test_preds] # Changed this line

    f1 = compute_f1_score(test_labels, predicted_labels)
    return f1

def evaluate_accuracy(index, test_embeddings, test_labels, init_data_labels, k, labels_range):
    test_preds = get_pred_vectors(index, test_embeddings, init_data_labels, k, labels_range)
    predicted_labels = [np.argmax(lst) for lst in test_preds]

    correct_predictions = sum(p == gt for p, gt in zip(predicted_labels, test_labels))
    accuracy = correct_predictions / len(test_labels)
    return accuracy

# keep the iteration with the best accuracy performance
def compare_iterations(list1, list2,best_k,k):
    # Check if both lists are empty
    if not list1 and not list2:
        return None  # Return None if both lists are empty

    # Get the maximum element from each list, using -inf as a fallback if a list is empty
    max_list1 = max(list1, default=float('-inf'))
    max_list2 = max(list2, default=float('-inf'))

    # Compare the maximum elements and return the corresponding list
    if max_list2 > max_list1:
        return list2, k  # Update to list2 if it has a higher max accuracy

    return list1, best_k  # Otherwise, keep list1 as the best

def compare_AL_iterations(list1, list2,best_k,k,best_weights,weights):
    # Check if both lists are empty
    if not list1 and not list2:
        return None  # Return None if both lists are empty

    # Get the maximum element from each list, using -inf as a fallback if a list is empty
    max_list1 = max(list1, default=float('-inf'))
    max_list2 = max(list2, default=float('-inf'))

    # Compare the maximum elements and return the corresponding list
    if max_list2 > max_list1:
        return list2, k, weights  # Update to list2 if it has a higher max accuracy

    return list1, best_k, best_weights  # Otherwise, keep list1 as the best

## active learning methods weight permutations with steps of 0.1
def generate_hyperparameter_permutations(step=0.1):
    # Step size for each hyper-parameter
    values = np.arange(0, 1.1, step)
    permutations = []

    # Nested loops for the three hyper-parameters
    for entropy_weight in values:
        for least_confident_weight in values:
            margin_weight = 1 - entropy_weight - least_confident_weight
            if 0 <= margin_weight <= 1:  # Ensure the sum is 1 and margin_weight is valid
                permutations.append((round(entropy_weight, 1),
                                     round(least_confident_weight, 1),
                                     round(margin_weight, 1)))

    return permutations


def simulation(
    methods,                       # List of initialization methods
    k_values,                      # List of k values
    AL_weights,                    # List of active learning weights permutations
    evaluate_performance,             # Function to evaluate performance
    init_data_embeddings,          # Initial data embeddings
    init_data_labels,              # Initial data labels
    sample_pool_embeddings,        # Sample pool embeddings for active learning
    sample_pool_labels,            # Sample pool labels for active learning
    test_embeddings,               # Test embeddings
    test_labels,                   # Test labels
    embedding_dim,                 # Dimension of the embeddings
    labels_range,                  # num of labels
    num_steps=25,                   # Number of steps (default is 25)
    samples_per_iter = 150              # Number of samples per iteration
):
    # Initialize lists to store accuracies and configurations
  best_setups = []  # To store step scores and config settings for active learning
  best_random_setups = []  # To store step scores and config settings for random sampling

  # Main loop
  for init_method in methods:
      best_setup_AL_performance = [0.05]  # To store accuracy scores for all steps for this combination
      best_setup_random_performance = [0.05]  # To store random accuracy scores for all steps for this combination
      best_AL_k = 0
      best_random_k = 0

      # Loop over k values
      for k in k_values:

          best_AL_weights = AL_weights[0]
          # Initialize FAISS random index setup data
          init_data_embeddings_random = copy.deepcopy(init_data_embeddings)
          init_data_labels_random = copy.deepcopy(init_data_labels)
          sample_pool_embeddings_random = copy.deepcopy(sample_pool_embeddings)
          sample_pool_labels_random = copy.deepcopy(sample_pool_labels)
          current_setup_random_performance = []  # To store random accuracy scores for current iteration

          # Counter to track no improvement in random performance
          no_improvement_random_steps = 0
          last_random_metric = -float('inf')

          # Random sampling index pipeline
          index_random = init_method(np.array(init_data_embeddings_random), embedding_dim)  # Init random here because no dependency on k
          for step in range(num_steps):
              if no_improvement_random_steps >= 5:
                  print(f"Early stopping RANDOM at step {step} for method {init_method.__name__}, k: {k}")
                  break

              # Random sampling: Select random samples
              selected_indices_random = select_random_samples(len(sample_pool_embeddings_random), samples_per_iter)

              # Transfer random samples to the ANN index
              sample_pool_embeddings_random, sample_pool_labels_random, init_data_embeddings_random, init_data_labels_random, index_random = transfer_samples(
                  sample_pool_embeddings_random, sample_pool_labels_random, init_data_embeddings_random, init_data_labels_random, selected_indices_random, index_random)

              # Evaluate random index performance this step
              random_metric = evaluate_performance(index_random, test_embeddings, test_labels, init_data_labels_random, k, labels_range)

              # Check if random metric improved
              if random_metric > last_random_metric:
                  no_improvement_random_steps = 0
              else:
                  no_improvement_random_steps += 1

              last_random_metric = random_metric

              if step % 6 == 0:
                  print(f"\n RANDOM Running step {step + 1} out of {num_steps} for method: {init_method.__name__}, k: {k}, {evaluate_performance.__name__}: {random_metric}")

              # Track random accuracy
              current_setup_random_performance.append(random_metric)

          # Compare and store the best k for random setup
          best_setup_random_performance, best_random_k = compare_iterations(best_setup_random_performance, current_setup_random_performance, best_random_k, k)

          ####################### Active learning pipeline

          for i, weights in enumerate(AL_weights[::-1]):

              # Initialize FAISS AL index setup data
              init_data_embeddings_AL = copy.deepcopy(init_data_embeddings)
              init_data_labels_AL = copy.deepcopy(init_data_labels)
              sample_pool_embeddings_AL = copy.deepcopy(sample_pool_embeddings)
              sample_pool_labels_AL = copy.deepcopy(sample_pool_labels)
              current_setup_AL_accuracies = []  # To store AL accuracy scores for current setup

              # Counter to track no improvement in AL performance
              no_improvement_AL_steps = 0
              last_AL_metric = -float('inf')

              # AL index pipeline
              AL_index = init_method(np.array(init_data_embeddings_AL), embedding_dim)
              for step in range(num_steps):
                  if no_improvement_AL_steps >= 5:
                      print(f"Early stopping AL at step {step} for method {init_method.__name__}, k: {k}, weights: {weights}")
                      break

                  # Get initial predictions
                  preds = get_pred_vectors(AL_index, sample_pool_embeddings_AL, init_data_labels_AL, k, labels_range)

                  # Select samples using active learning
                  selected_indices = select_samples(preds, samples_per_iter, entropy_weight=weights[0], least_confident_weight=weights[1], margin_weight=weights[2])

                  # Transfer samples to the ANN index
                  sample_pool_embeddings_AL, sample_pool_labels_AL, init_data_embeddings_AL, init_data_labels_AL, AL_index = transfer_samples(
                      sample_pool_embeddings_AL, sample_pool_labels_AL, init_data_embeddings_AL, init_data_labels_AL, selected_indices, AL_index)

                  # Evaluate accuracy for AL-selected samples
                  AL_metric = evaluate_performance(AL_index, test_embeddings, test_labels, init_data_labels_AL, k, labels_range)

                  # Check if AL metric improved
                  if AL_metric > last_AL_metric:
                      no_improvement_AL_steps = 0
                  else:
                      no_improvement_AL_steps += 1

                  last_AL_metric = AL_metric

                  if (step%6==0): print(f"metric for step {step + 1}, weights {weights}: {AL_metric}")

                  # Track accuracy
                  current_setup_AL_accuracies.append(AL_metric)

              # Compare and store the best active learning accuracies
              best_setup_AL_performance, best_AL_k,best_AL_weights = compare_AL_iterations(best_setup_AL_performance, current_setup_AL_accuracies, best_AL_k, k,best_AL_weights,weights)

      # Store the best accuracies and configurations
      best_setups.append((best_setup_AL_performance, init_method.__name__, best_AL_k, best_AL_weights))
      best_random_setups.append((best_setup_random_performance, init_method.__name__, best_random_k))

  return best_setups, best_random_setups


####### MAIN

 #Generate all hyper-parameter permutations with steps of 0.1
AL_weights = generate_hyperparameter_permutations()
methods = [init_L2, init_IVF, init_LSH,init_HNSW]


# load mmnist fashion data
df = pd.read_csv("/content/fashion-mnist_train.csv")
pixel_columns = [col for col in df.columns if 'pixel' in col]
MMNIST_labels = df['label'].values
MMNIST_embeddings_array = df[pixel_columns].values  # Get the pixel columns as a NumPy array
MMNIST_samples_per_iter = 840 ## up to 50% of data
MMNIST_labels_range = [0,1,2,3,4,5,6,7,8,9]
MMNIST_embedding_dim = MMNIST_embeddings_array.shape[1]
MMNIST_k_values = [10]
# Split MMNIST data
init_data_labels, init_data_embeddings, sample_pool_labels, sample_pool_embeddings, test_labels, test_embeddings = split_data(MMNIST_labels, MMNIST_embeddings_array,init_data_size=800, test_size=9000)
best_MMNIST_setups, best_MMNIST_random_setups = simulation(
        methods=methods,
        k_values=MMNIST_k_values,
        AL_weights=AL_weights,
        evaluate_performance=evaluate_F1,  # Assuming this function evaluates the accuracy
        init_data_embeddings=init_data_embeddings,
        init_data_labels=init_data_labels,
        sample_pool_embeddings=sample_pool_embeddings,
        sample_pool_labels=sample_pool_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        embedding_dim=MMNIST_embedding_dim,
        labels_range=MMNIST_labels_range,
        num_steps=25,
        samples_per_iter=MMNIST_samples_per_iter
    )

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
TRIVIAL_labels_range = [0, 1, 2]
TRIVIAL_samples_per_iter = 100
TRIVIAL_k_values = [3]
# Split TEST data
init_data_labels, init_data_embeddings, sample_pool_labels, sample_pool_embeddings, test_labels, test_embeddings = split_data(TRIVIAL_labels, TRIVIAL_embeddings_array,init_data_size=200, test_size=4400)

## second simulation

best_TRIVIAL_setups, best_TRIVIAL_random_setups = simulation(
        methods=methods,
        k_values=TRIVIAL_k_values,
        AL_weights=AL_weights,
        evaluate_performance=evaluate_accuracy,  # Assuming this function evaluates the accuracy
        init_data_embeddings=init_data_embeddings,
        init_data_labels=init_data_labels,
        sample_pool_embeddings=sample_pool_embeddings,
        sample_pool_labels=sample_pool_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        embedding_dim=TRIVIAL_embedding_dim,
        labels_range=TRIVIAL_labels_range,
        num_steps=25,
        samples_per_iter=TRIVIAL_samples_per_iter
    )

## mminst augmented biased dataset loading                                                                                                                             
df = pd.read_csv("/content/fashion-mnist_train.csv")
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

# Create 3 times more class 2 vectors
num_class_2_to_generate = len(class_2_vectors) * 2  # we want to triple the count, so we need to generate 2 additional for each
new_class_2_vectors = np.array([add_gaussian_noise(vec) for vec in class_2_vectors for _ in range(3)])[:num_class_2_to_generate]

# Combine the new dataset
AUGMENTED_embeddings_array = np.vstack((new_class_2_vectors, class_2_vectors, class_4_vectors))
AUGMENTED_labels = np.array([2] * len(new_class_2_vectors) + [0] * len(class_2_vectors) + [1] * len(class_4_vectors))#

AUGMENTED_samples_per_iter = 240 ## up to 50% of data
AUGMENTED_labels_range = [0,1]
AUGMENTED_embedding_dim = MMNIST_embeddings_array.shape[1]
AUGMENTED_k_values = [2]
# Split MMNIST data
init_data_labels, init_data_embeddings, sample_pool_labels, sample_pool_embeddings, test_labels, test_embeddings = split_data(AUGMENTED_labels, AUGMENTED_embeddings_array,init_data_size=480, test_size=2400)

## simulation 3
best_AUGMENTED_setups, best_AUGMENTED_random_setups = simulation(
        methods=methods,
        k_values=AUGMENTED_k_values,
        AL_weights=AL_weights,
        evaluate_performance=evaluate_F1,  # Assuming this function evaluates the accuracy
        init_data_embeddings=init_data_embeddings,
        init_data_labels=init_data_labels,
        sample_pool_embeddings=sample_pool_embeddings,
        sample_pool_labels=sample_pool_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        embedding_dim=AUGMENTED_embedding_dim,
        labels_range=AUGMENTED_labels_range,
        num_steps=25,
        samples_per_iter=AUGMENTED_samples_per_iter
    )

print("we are done")
