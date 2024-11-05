import json
import matplotlib.pyplot as plt

# Load the JSON files
with open('best_AUGMENTED_random_setups_final.json', 'r') as f:
    best_AUGMENTED_random_setups = json.load(f)

with open('best_AUGMENTED_setups2.json', 'r') as f:
    best_AUGMENTED_setups = json.load(f)

with open('best_MMNIST_random_setups_final.json', 'r') as f:
    best_MMNIST_random_setups = json.load(f)    

with open('best_MMNIST_setups2.json', 'r') as f:
    best_MMNIST_setups = json.load(f)

with open('best_TRIVIAL_random_setups_final.json', 'r') as f:
    best_TRIVIAL_random_setups = json.load(f)    

with open('best_TRIVIAL_setups2.json', 'r') as f:
    best_TRIVIAL_setups = json.load(f)

def plot_comparison(best_random_setup,best_AL_setup,data_name): 
### given a dataset, this function compares each random and AL setup for each method chosen

  # Plot each pair of symmetrical elements
  for i, (random_setup, AL_setup) in enumerate(zip(best_random_setup, best_AL_setup)):
      # Extract the necessary data
      random_accuracies = random_setup[0]  # List of accuracies
      augmented_accuracies = AL_setup[0]  # List of accuracies
      title = data_name + " dataset " + random_setup[1]  # Plot title
      weights = ' '.join(str(element) for element in AL_setup[3])
      legend_name = "AL with weights " + weights  # Legend name

      # Plot the accuracies
      plt.figure(figsize=(10, 6))
      plt.plot(random_accuracies, label='Random Setup')
      plt.plot(augmented_accuracies, label=legend_name)
      
      # Set plot title and labels
      plt.title(title)
      plt.xlabel('Step')
      plt.ylabel('Accuracy')
      plt.legend()
      
      # Show or save the plot
      plt.show()  # Or save with plt.savefig(f'plot_{i}.png')

def dataset_comparison(best_random_setup,best_AL_setup,data_name): 

  plt.figure(figsize=(12, 8))
  title = data_name + " dataset review"  # Plot title
  for i, (random_setup, AL_setup) in enumerate(zip(best_random_setup, best_AL_setup)):
      # Extract the necessary data
      random_accuracies = random_setup[0]  # List of accuracies
      augmented_accuracies = AL_setup[0]  # List of accuracies
      weights = ' '.join(str(element) for element in AL_setup[3])
      legend_name = random_setup[1] +" AL with weights " + weights  # Legend name
      plt.plot(random_accuracies, label=random_setup[1] + ' Random Setup')
      plt.plot(augmented_accuracies, label=legend_name)
      
  # Set plot title and labels
  plt.title(title)
  plt.xlabel('Step')
  plt.ylabel('Accuracy')
  plt.legend()
  
  # Show or save the plot
  plt.show()



#plot_comparison(best_AUGMENTED_random_setups,best_AUGMENTED_setups,"Augmented")
#plot_comparison(best_MMNIST_random_setups,best_MMNIST_setups,"MMNIST")
#plot_comparison(best_TRIVIAL_random_setups, best_TRIVIAL_setups,"TRIVIAL")

dataset_comparison(best_AUGMENTED_random_setups,best_AUGMENTED_setups,"Augmented")
dataset_comparison(best_MMNIST_random_setups,best_MMNIST_setups,"MMNIST")
dataset_comparison(best_TRIVIAL_random_setups, best_TRIVIAL_setups,"TRIVIAL")
