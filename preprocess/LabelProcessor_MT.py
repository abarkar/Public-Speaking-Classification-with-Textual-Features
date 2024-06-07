"""
This script creates classes of low and high quiality of performance based on
the subjective ratings provided in the dataset.

@author: Alisa Barkar, alisa.george.barkar@gmail.com
"""
# Data & Mathmatics
import pandas as pd
# System
import sys
import os
# Configuration
from config_reader import read_config
# Global variable declaration
global rootDirPath, dataset, dimension, clip, model, clasSeparator, scoreType


def readData():
    """
    Read and cleans data.
    Parameters:
    rootDirPath (str): Directory containing the dataset named "{dataset}_aggregated_scores.csv"
    
    Returns:
    data_labels (DataFrame): Slice of initial data with considered clip, scoreType and dimensions.
    dim_columns (list): list of column names of considered dimensions of dataset.
    index (list): list of sample IDs.
    """
    # Data directory
    data_dir=os.path.join(rootDirPath, "data", dataset, dataset+"_aggregated_ratings.csv")
    # Read data: 0 column should contain IDs of data samples
    data_labels= pd.read_csv(data_dir, sep=",")#, index_col=0)
    # Take a clip slice of dataset if needed
    data_labels = data_labels[(data_labels['clip'] == clip) & (data_labels['aggregationMethod'] == aggregationMethod)]
    # Take the dimension'
    # if dimension == "all":
    #     # Columns to exclude
    #     columns_to_exclude = ['ID', 'clip', 'aggregationMethod']
    #     # Select all columns except the specified ones
    #     dim_columns= data_labels.drop(columns=columns_to_exclude).columns
    # else:
    #     dim_columns= dimension
    index = data_labels.index

    for dim in dimension:
        # # Make sure that scores will be read as float
        # # Convert the specified columns
        # print(data_labels[dim].head())
        # data_labels[dim] = data_labels[dim].apply(lambda col: col.str.replace(',', '.').astype(float))

        # Ensure the columns are of float type
        data_labels[dim] = data_labels[dim].astype(float)

    return data_labels, index




def extractConfidenceLabels():
    """
    Categorize data based on the given method.
    
    Parameters:
    data_labels (DataFrame): The input DataFrame that is downlodaded via the function readData().
    dim_columns (list of strings or string): The dimension columns that will be categorized.
    scoreType (str): The method to use for categorization ('mean', 'median', 'Q1Q3').
    
    Returns:
    DataFrame: The DataFrame with a new column for categorization.
    """
    
    # Read data 
    data_labels, index = readData()
    # Go through all the dimensions and for each create the labels
    for dim in dimension:
        print("Considered dimension is: ", dim, "\n")
        # Create thresholds for data class separation
        threshold = [0, 0]
        if clasSeparator == 'mean':
            threshold[0] = data_labels[dim].mean()
            threshold[1] = data_labels[dim].mean()
        elif clasSeparator == 'median':
            threshold[0] = data_labels[dim].median()
            threshold[1] = data_labels[dim].median()
        elif clasSeparator == 'Q1Q3':
            threshold[0] = data_labels[dim].quantile(0.25)
            threshold[1] = data_labels[dim].quantile(0.75)
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'median', 'Q1Q3'.")
        
        # Print dimension
        print("Thresholds are: ", threshold[0], " and ", threshold[1], "\n")

        # Create the new DataFrame for classes labels storage
        data_classified = pd.DataFrame()
        for index, row in data_labels.iterrows():   
            if row[dim] >= threshold[0]:
                data_classified = data_classified._append({'ID': row["ID"], 'label': 1}, ignore_index=True)
            if row[dim] < threshold[1]:
                data_classified = data_classified._append({'ID': row["ID"], 'label': 0}, ignore_index=True)

        # Check classes sizes
        class_1 = len(data_classified.loc[data_classified['label'] == 1])
        class_0 = len(data_classified.loc[data_classified['label'] == 0])
        print("Nb of samples in class_1: ", class_1, "Nb of samples in class_0: ", class_0, "\n")
        print("Nb of samples of the dataset: ", data_classified['label'].value_counts())
        # Saving directory
        save_dir=os.path.join(rootDirPath, "data", dataset, "labels", clip)
        # Check if the directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Save data with classes separation to the .csv
        data_classified.to_csv(f'{save_dir}/{dim}Label.csv', index=False)

if __name__ == "__main__":
    """
    This programm categorize data w.r.t. the chosen thresholding.
    
    Parameters:
    rootDirPath : Directory of the data_labels.csv file with the scores of different dimensions. 
                    Data has to be 2D array of N x (D+4) dimensions where N -- number of samples in the dataset
                    D -- number of dimensions that are annotated in the dataset (i.e. persuasiveness, confidence, etc.)
                    3 represent 3 more columns that are not dataset dependent and has to be specyfied in the data: 
                    ID (ids of samples in dataset), video(clip type i.e. full, beg, mid, end), scoreType(types 
                    of annotation agregation i.e. mean, rms, etc.).
    dataset: The name of the used dataset (directories have to be named the same way).
    dimension: The dimension from dataset interesting for the analysis (all, persuasiveness, engagement, etc.).
    clip: The type of considered video slices (full, beginning, middle, end).
    model: Classification model (NOT USED HERE)
    clasSeparator: The value that will be used as the threshold for separation on classes ('mean', 'median', 'Q1Q3').
    scoreType: The type of code aggregation that was used for obtaining the scores (mean, rms, etc.)
    
    Returns:
    DataFrame: The DataFrame of N x 2 dimensions where the first column represents sample IDs in the dataset and 
                the second column has 1/0 labels.
    """
    # Now the new version automatically take the values from the configuration file via the config_reader.py
    # Read the configuration
    config = read_config()


    # Use the configuration values
    rootDirPath = config['rootDirPath']
    dataset = config['dataset']
    dimension = config['dimension']
    clip = config['clip']
    clasSeparator = config['clasSeparator']
    aggregationMethod = config['aggregationMethod']

    # Example usage in your script
    print(f"Root Directory Path: {rootDirPath}")
    print(f"Dataset: {dataset}")
    print(f"Dimension: {dimension}")
    print(f"Clip: {clip}")
    print(f"Class Separator: {clasSeparator}")
    print(f"Aggregation Method: {aggregationMethod}")

    # Adding the rootDirPath to the system path
    sys.path.append(rootDirPath)
    # Execute the Separation on Classes
    extractConfidenceLabels()