# TECHNION_LAB2_PROJECT
integration of active learning in ANN environment using KNN as performance evaluation:
# Overview:
Our project tackles the question whether a smart selection of samples to agregate into ANN indexes, utilizing Active Learning strategies, can increase the performance of class predictive models in a sparse data environment. To answer this question, we developed a pipeline, which simutanieously feeds random selected samples and AL selected samples into the same ANN method,utilizing K-nearest neighbors (KNN) for the prediction task. The analysis of the results suggest that AL stategies could be impactful to performance of predictive KNN models.

# System Description:
Our project is divided into 3 parts: 1) the visualization of the data in order to assess it's charicharistics. this part can be shown in the "PCA_vizualizations" folder. 2) the pipeline simulation, which can be viewed in the "PIPELINE_SIMULATION.py" file. 3) the analysis of the simulation can be seen at the "results_analysis.py" file

# Getting Started:
## Setting up the Development Environment:
Pre-requisites: Git and Anaconda. 
download the initial datasets from the drive link here: 
https://drive.google.com/drive/folders/11GtaGzPMuIKdhc9F0ixAvQB6oAq63W4c?usp=sharing
in each file at the start of the main section, change the file path of uploaded files(for example the datasets or the result files) to their updated location.
## files paths required for set up of each file: 
### PCA_vizualizations:
"AUGMENTED_PCA_VIZ.py" and "MMNIST_PCA_VIZ.py" require the "fashion-mnist_train.csv" from the drive link above.
"TRIVIAL_PCA_VIZ.py" requires the "index.pkl" from the drive 
"PIPELINE_SIMULATION.py" requires both "fashion-mnist_train.csv" and "index.pkl"
"results_analysis.py" requires all the files from the "results_files" directory in this project repository.
To install and run the code on your local machine, follow these steps:
1. ### Clone the repository
   First, clone the repository to your local machine using Git. Open a terminal and run the following command:
    ```bash
    git clone https://github.com/AmitLandau-collab/TECHNION_LAB2_PROJECT
2. ### Create and activate the conda environment
   After cloning the repository, navigate into the project directory:
    ```bash
    cd TECHNION_LAB2_PROJECT
    ```
    Then, use the following command to create a conda environment from the environment.yml file provided in the project:
    ```bash
    conda env create -f environment.yml
    ```
    Activate the environment with the following command:
   ```bash
    conda activate project_env
    ```


