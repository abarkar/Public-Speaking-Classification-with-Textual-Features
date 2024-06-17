"""
This script trains and tests classical ML models based on the different multimodal features.

The configuration is downloaded from the configuration.ini file. It choses:
    rootDirPath,
    dataset,
    dimension,
    clip,
    model,
    clasSeparator,
    scoreType,
    task

@author: Alisa Barkar, alisa.george.barkar@gmail.com
"""

# Data & Mathmatics
import pandas as pd
import numpy as np
import seaborn as sns
import random
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from itertools import repeat, chain
# Feature importance analysis
from feedback.SHAP import ModelInterpret#, setGlobal
import scipy.stats as st
from feedback.feedback_generator import feedback
from feedback.SHAP import grouped_shap
# Trainins/test
from sklearn.model_selection import train_test_split, LeaveOneOut
from models.ML_Model import classificator #definedSVM, definedRFC, definedLR, SupportVectorMachine, RandomForest, LogRegression#, simpleDNN
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# Save results/models
import json
from joblib import dump,load
import matplotlib.pyplot as plt
# System
import sys
import os
import glob
import warnings
warnings.filterwarnings("ignore") # depricates warnings
# Import the config reader
from config_reader import read_config
# Global
global rootDirPath
global dataset
global dimension
global clip
global model
global task
global modalities
global threshold
global featureSelection



def feature_selection(X, Y, target):
    """
    Feature selection based on Spearman correlation. 
    Save the new X and target to the same files as before.

    Parameters:
    X (DataFrame): feature values.
    Y (DataFrame): labels.

    Return:
    X (DataFrame): only features with the absolute value of correlation bigger than threshold.
    """
    # Align features and labels along the IDs.
    data = X.join(Y, how='left')
    # Calculate Spearman correlation
    corr = data.corr(method='spearman')
    # Threshold data w.r.t. correlation
    thresholded_corr = corr[abs(corr['label']) > threshold]
    # TODO: check why -2 index and if it is because "ID" and "label" then add these columns explicitly
    X = X[thresholded_corr.index[:-2]]
    # Set tagret data
    if (dataset == "POM"):
        target = X.loc[[136647]]
    elif (dataset == "MT"):
        target = X.loc[['BOD09']]
    # Save target and background data 
    bachground_dir = os.path.join(rootDirPath, "demo", dataset, "background.csv")
    target_dir = os.path.join(rootDirPath, "demo", dataset, "target.csv")
    # Save target and background data
    X.to_csv(bachground_dir)
    target.to_csv(target_dir)
    return X





def dataPreprocessing(X, Y):
    """
    Checks that X and Y have the same list of IDs. 
    Saves target and background data for further SHAP analysis.

    Parameters:
    X (DataFrame): feature values.
    Y (DataFrame): labels.

    Return:
    X (DataFrame): feature values.
    Y (DataFrame): labels.
    target (DataFrame): One chosen sample of dataset
    """
    # Check for the same IDs in X(features) and Y(labels)
    # print(len(Y.index))
    # print(len(X.index))
    Y = Y.loc[(Y["ID"]).isin(X["ID"])]
    X = X.loc[(X["ID"]).isin(Y["ID"])]

    # Set tagret data
    if (dataset == "POM"):
        # target = X.loc[[136647]]
        target = X[X['ID'] == 136647]
    elif (dataset == "MT"):
        # target = X.loc[['BOD09']]
        target = X[X['ID'] == 'BOD09']
    # Save target and background data 
    bachground_dir = os.path.join(rootDirPath, "demo", dataset, "background.csv")
    target_dir = os.path.join(rootDirPath, "demo", dataset, "target.csv")
    # Save target and background data
    X.to_csv(bachground_dir)
    target.to_csv(target_dir)
    return X, Y, target



def loadFeturesByCategories(feature_dir):
    """
    Loads features from the .csv files. 

    Parameters:
    feature_dir (string): directory containing feature files.

    Return:
    feature_df (DataFrame): feature values.
    group_by_category (dictionary): dictoanary of pairs (category_name:list_of_features).
    """
    # Create dataset of feature:
    feature_df=pd.DataFrame()
    # Create dictionaries of feature categories
    group_by_category={}
    # Go through all considered modalities
    for mod in modalities:
        mod_dir=os.path.join(feature_dir, mod)
        for feature_file in glob.glob(os.path.join(mod_dir, '*.csv')):
            # File with features
            feat_dir=os.path.join(mod_dir, feature_file)
            # Extract feature category name
            clean_name = os.path.splitext(os.path.basename(feature_file))[0]
            # Download features from .csv file
            df = pd.read_csv(feat_dir)
            if 'ID' not in df.columns:
                raise KeyError(f"'ID' column is missing in file {feature_file}")
            # Extract names of features
            feature_names =df.drop(columns=["ID"]).columns
            # Create the new pair of (category_name: list_of_features)
            group_by_category[str(clean_name)] = list(feature_names)
            # Add features to the big feature dataset aligning along IDs
            # Add features to the big feature dataset aligning along IDs
            print(f'{clean_name} features:', len(df.index))
            if feature_df.empty:
                feature_df = df
            else:
                merged_df = pd.merge(feature_df, df, on="ID", how='inner', indicator=True)
                mismatched_rows = merged_df[merged_df['_merge'] != 'both']
                if not mismatched_rows.empty:
                    print("Mismatched rows:", mismatched_rows)
                feature_df = merged_df.drop(columns='_merge')
            # print(feature_df.columns)
            print('full feature set: ', len(feature_df.index))
            # Check for duplicate IDs
            if feature_df['ID'].duplicated().any():
                feature_df = feature_df.drop_duplicates(subset=['ID'])
    feature_df.to_csv(f'{feature_dir}/full_feature_set.csv')
    return feature_df, group_by_category




def loadRatings(dim):
    """
    Loads labels from the .csv file w.r.t. the task (classification/regression). 

    Return:
    labels (DataFrame): label values.
    """
    label_file = os.path.join(rootDirPath, "data", dataset, "labels", clip, dim + "Label.csv")
    if task =="classification":
        labels=pd.read_csv(label_file)    
    else:
        #TODO: adapt the function for the regression task
        labels=pd.DataFrame()
    return labels


def calculate_ci(data):
    """
    Calculate mean and confidence interval from the data in provided list. 

    Parameters:
    data (list): list of values.

    Return:
    mean (float): mean of the values in the data.
    ci (float): confidnce intervals of the mean of the values in the data.
    """
    mean = np.mean(data)
    ci = st.norm.interval(confidence=0.95, loc=mean, scale=st.sem(data))
    return mean, ci


def choseBestParametersForClassification(X, Y, clf, output_dir):
    """
    Grid search for the given model. 

    Parameters:
    X (DataFrame): list of features.
    Y (DataFrame): list of labels.
    clf (classifier object): classification model.
    output_dir (string): directory to save data.

    Return:
    rf_param (list): list of the best parameters.
    best_model (classifier): the best model after grid search.
    """
    
    
    # Split data for training and testing for the grid search
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
    # USe Grid search with the parameters specifies in the classifer class
    rf_param, rf_train, rf_test, best_model = clf.GridSearch(X_train, y_train, X_test, y_test)
    # Save the best parameters
    os.makedirs(output_dir, exist_ok=True)
    txtForSave = open(f'{output_dir}/best_parameters_rf.txt', 'w')
    txtForSave.write("best parameters: {}".format(rf_param) +"\n" + " best_train_score: {}".format(rf_train) +"\n" + " best_test_score: {}".format(rf_test) +"\n"  )
    txtForSave.close()
    
    return rf_param, best_model



def averageF1Score(X, Y, best_param, clf, output_dir):
    """
    Calculation of the average F1 score based on the model with the best parameters. 

    Parameters:
    X (DataFrame): list of features.
    Y (DataFrame): list of labels.
    best_param (list): list of the best parameters.
    clf (classifier object): classification model.
    output_dir (string): directory to save data.
    """
    # Avegare F1 Score over 50 trainings
    test_mean = []
    for i in range(0, 50):
        # Split the data with the new seed
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        # Obtain f1 score for the model with the best parameters
        rf_test, f1_score = clf.Defined(X_train, y_train, X_test, y_test, best_param)
        test_mean.append(f1_score)
    # Calculate average F1 score 
    mean, ci = calculate_ci(test_mean)
    # Save average F1 score
    txtForSave = open(f'{output_dir}/F1_score.txt', 'w')
    txtForSave.write("mean: {}".format(mean) + " confidence interval: {}".format(ci) + "\n")
    txtForSave.close()




def randomClassifier(X, Y, output_dir):
    """
    Random classification and all 1 predition (to compare accuracy with the classififers). 
    """
    # Launch random classification 300 times
    test_mean = []
    for i in range(0, 300):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i, stratify=Y)
        # Initialize random prediction
        random_list  = random.choices(range(0, 2), k = len(y_test.index))
        # Initialise prediction with all 1
        ones_array = np.ones(len(y_test.index))
        # F1 for random prediction
        f1_rand = f1_score(y_test['label'], random_list)
        # F1 for all 1 prediction
        f1_ones = f1_score(y_test['label'], ones_array)
        # accuracy for all 1 prediction
        acc_rand = accuracy_score(y_test['label'], random_list)
        # accuracy for all 1 prediction
        acc_ones = accuracy_score(y_test['label'], ones_array)
        # Add f1 and accuracies
        test_mean.append([f1_rand, f1_ones, acc_rand, acc_ones])
    for i in range(4):
        # Claculate mean and confidential interbal for the metric
        mean, ci = calculate_ci(test_mean[:][i])
        # Make sure that output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Save results to the output directory 
        txtForSave = open(f'{output_dir}/random_f1_score.txt', 'w')
        txtForSave.write("mean: {}".format(mean) + " confidence interval: {}".format(ci) + "\n")
        txtForSave.close()



def shapAnalysis(best_model, X_train, X_test, group_by_category, output_dir):
    """
    SHAP analysis. 

    Parameters:
    best_model (classifier object): classification model with the best parameters.
    X_train (DataFrame): list of features for training.
    X_test (DataFrame): list of features for testing.
    group_by_category (dict): dictionary of pairs (categor:list_of_features).
    output_dir (string): directory to save data.
    """
    # TODO: check the use and whether we need it here
    feedback_generator = feedback(f'{output_dir}')
    # group_shap = grouped_shap(svm, X_test, target, group_by_category, feature_name)
    # Extract feature_names
    feature_name = X_train.columns
    # Calculate SHAP by categories of features with the best model
    group_shap = grouped_shap(best_model, X_train, X_test, group_by_category, feature_name)
    # feedback_generator.ABS_SHAP(group_shap)

    # Interpret Model output with the features
    ModelInterpret(best_model, X_test, group_by_category, feature_name)


def leaveOneOutTrain(X, Y, best_param, clf, output_dir):
    """
    Leave-one-out method for testing the model with the best parameters. 

    Parameters:
    best_model (classifier object): classification model with the best parameters.
    X (DataFrame): list of features.
    Y (DataFrame): list of labels.
    clf (classifier object): classification model.
    output_dir (string): directory to save data.
    """
    # Initialisation of leave one out object
    loo = LeaveOneOut()
    # Go through the data batches for each left out ID
    test_mean = []
    for i, (train_index, test_index) in enumerate(loo.split(X, Y)):
        # Take train and test data and labels according to the indexes of the current batch
        X_train = X.loc[X.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        y_train = Y.loc[Y.index[train_index]]
        y_test = Y.loc[Y.index[test_index]]
        # Accaracy of the model
        rf_test, f1_score = clf.Defined(X_train, y_train, X_test, y_test, best_param)
        # Save the accuracy
        test_mean.append(rf_test)
    # Calculate mean accuracy 
    mean, ci = calculate_ci(test_mean)
    # Save results
    txtForSave = open(f'{output_dir}/lvo_accuracy_score.txt', 'w')
    txtForSave.write("mean: {}".format(mean) + " confidence interval: {}".format(ci) + "\n")
    txtForSave.close()


def save_correlations_to_csv(X, Y, output_dir):
    """
    Calculate Spearman and Pearson correlations between features and labels, save results to .csv. 

    Parameters:
    X (DataFrame): list of features.
    Y (DataFrame): list of labels.
    output_dir (string): directory to save data.
    """
    data = pd.merge(X, Y, on='ID', how='inner')#.drop('ID')
    spearman_corr = data.select_dtypes(include=[float, int]).corr(method='spearman')['label'].drop('label')
    pearson_corr = data.select_dtypes(include=[float, int]).corr(method='pearson')['label'].drop('label')
      
    correlation_df = pd.DataFrame({
        'feature': spearman_corr.index,
        'spearman': spearman_corr.values,
        'pearson': pearson_corr.values
    })
    correlation_df.to_csv(f"{output_dir}/feature_label_correlations.csv", index=False)

def plot_feature_correlations(X, output_dir):
    """
    Calculate Spearman and Pearson correlations between features. Saves results as correlation matrix.

    Parameters:
    X (DataFrame): list of features.
    output_dir (string): directory to save data.
    """
    spearman_corr = X.select_dtypes(include=[float, int]).corr(method='spearman')
    pearson_corr = X.select_dtypes(include=[float, int]).corr(method='pearson')
    
    plt.figure(figsize=(10, 8))
    plt.title("Spearman Correlation Matrix")
    plt.imshow(spearman_corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.savefig(f"{output_dir}/spearman_feature_correlation.png")
    plt.clf()

    plt.figure(figsize=(10, 8))
    plt.title("Pearson Correlation Matrix")
    plt.imshow(pearson_corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.savefig(f"{output_dir}/pearson_feature_correlation.png")
    plt.clf()

def revert_dict(d):
    """
    Revert the dictionary. 

    Parameters:
    d (dictionary): dictionary with pairs (category:list_of_features).
    
    Return:
    dictionary: {(feature_name:category)}
    """
    return dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))

def calc_corr_by_categories(corr, group_by_category, output_dir, method):
    """
    Calculate correlation between features and labels and then group it by the categories. 
    For each category plot the boxplot.

    Parameters:
    
    corr (array): correlations between features and labels. 
    group_by_category (dict): dictionary with pairs (category:list_of_features).
    output_dir (string): directory to save data.
    type (string): type of correlation.
    """
    groupmap = revert_dict(group_by_category)
    # Convert the Series to a DataFrame
    corr_df = pd.DataFrame(data=corr).reset_index()
    # Rename the columns to 'feature' and 'corr'
    corr_df.columns = ['feature', 'corr']
    corr_df['group'] = corr_df['feature'].map(groupmap).values
    grouped = corr_df.groupby('group')['corr'].apply(list)
    
    plt.figure(figsize=(12, 6))
    plt.title(f"{method.capitalize()} Correlation Distribution by Categories")
    plt.boxplot(grouped, labels=grouped.index)
    plt.xticks(rotation=25, fontsize='x-small')
    plt.ylabel(f"{method.capitalize()} Correlation")
    plt.savefig(f"{output_dir}/{method}_correlation_by_category.png")
    plt.clf()
    # Define a function to calculate the mean of each list
    def list_mean(lst):
        return sum(lst) / len(lst)

    # Apply the function to each element of the series
    grouped = grouped.apply(list_mean)
    grouped.to_csv(f"{output_dir}/{method}_mean_corr_by_category.csv", header=[f'mean_{method}_corr'])

def correlation_analysis(X, Y, group_by_category, output_dir):
    """
    Launch calculation of different correlations.

    Parameters:
    X (DataFrame): list of features.
    Y (DataFrame): list of labels. 
    group_by_category (dict): dictionary with pairs (category:list_of_features).
    output_dir (string): directory to save data.
    """
    corr_dir = os.path.join(output_dir, "correlation")
    # Ensure the output directory exists
    os.makedirs(corr_dir, exist_ok=True)

    # Calculate and save correlations between features and labels
    save_correlations_to_csv(X, Y, corr_dir)

    # Calculate and plot correlations between all features in X
    plot_feature_correlations(X, corr_dir)

    # Join features and labels for correlation analysis
    data =  pd.merge(X, Y, on='ID', how='inner')
    
    # Calculate and save category-wise correlations
    for method in ['spearman', 'pearson']:
        corr = data.select_dtypes(include=[float, int]).corr(method=method)['label'].drop('label')
        corr = corr.fillna(0)
        calc_corr_by_categories(corr, group_by_category, corr_dir, method)




def mainPipeline():
    """
    Main function that executes the pipeline steps:
    1. Download features, preprocess, select features;
    2. Correlation analysis;
    3. Grid Search for the best parameters;
    4. Leave One Out for model testing;
    5. SHAP analysis;
    """
    # Now dimensions are already downloaded from the configuration file
    #dimentions = getDimentions(dataset)

    # Go through each considered dimension 
    for dim in dimension:
        output_dir=os.path.join(rootDirPath, "results", dataset, dim, clip)
        feature_dir = os.path.join(rootDirPath, "data", dataset, "features", clip)
        print("------------------" + dim + "------------------")
        print("*********************** Load Features ***********************")
        # rate_type = dim
        # TODO: rewrite SetGLobal in SHAP.py to use configuration file instead.
        # setGlobal(dataset, dim)

        # Load features of different modalities from the .csv files, create feature grouping by category
        X, group_by_category = loadFeturesByCategories(feature_dir)
        print(X)
        # Load Ratings w.r.t. task
        Y = loadRatings(dim)
        print(Y)
        # Check correspondance of IDs in features and labels, prepare target and background .csv for SHAP analysis
        # TODO: check whether we need target and background at all
        X, Y, target = dataPreprocessing(X, Y)
        print(target)
        print("*********************** Correlation Analysis with Labels ***********************")
        # correlation(X, Y, group_by_category)
        correlation_analysis(X, Y, group_by_category, output_dir)
        print("*********************** featureSelection ***********************")
        # X = feature_selection(X, Y)
        if (len(X.columns) > 0):
            data = pd.merge(X, Y, on="ID")
            Y = data[["ID", "label"]]
            X = data.drop(columns=["label"])
            X.set_index('ID', inplace=True)
            Y.set_index('ID', inplace=True)
            for clf_model in model:
                # Set the path to save model results
                model_res_dir = os.path.join(output_dir, clf_model)
                # Change the data indexing for the train/test splitting
                print("*********************** Create a classificator object ***********************")
                clf = classificator(clf_model)
                print("*********************** Chose Best Parameters ***********************")
                best_param, best_model = choseBestParametersForClassification(X, Y, clf, model_res_dir)
                print("*********************** LeaveOneOut ***********************")
                leaveOneOutTrain(X, Y, best_param, clf, model_res_dir)
                print("*********************** AverageF1 ***********************")
                averageF1Score(X, Y, best_param, clf, model_res_dir)
                print("*********************** Random ***********************")
                randomClassifier(X, Y, output_dir)
                print("*********************** SHAP ***********************")
                # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
                # best_model.fit(X_train, y_train.values.ravel())
                # shapAnalysis(best_model, X_train, X_test, group_by_category, model_res_dir)
                



if __name__ == "__main__":
    """
    Initialisation function that downloads values from the configuration file.
    """

    # Read the configuration
    config = read_config()

    # Use the configuration values
    rootDirPath = config['rootDirPath']
    dataset = config['dataset']
    dimension = config['dimension']
    clip = config['clip']
    model = config['model']
    task = config['task']
    modalities = config['modalities']
    threshold = config['threshold']
    featureSelection = config['featureSelection']

    # Example usage in your script
    print(f"Root Directory Path: {rootDirPath}")
    print(f"Dataset: {dataset}")
    print(f"Dimension: {dimension}")
    print(f"Clip: {clip}")
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Modalities: {modalities}")
    print(f"Threshold: {threshold}")
    print(f"Feature Selection: {featureSelection}")


    # Adding the rootDirPath to the system path
    sys.path.append(rootDirPath)
    # Launch main pipeline: Feature download, Feature selection & Fusion, Train/Test, Importance analysis
    mainPipeline()


