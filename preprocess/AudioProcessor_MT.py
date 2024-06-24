"""
The AudioProcessor_MT.py script extracts eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) features from MT dataset files and saves the resulting feature set to a CSV file.
"""


import sys
import os
import glob
import pandas as pd
import json
import opensmile
from tqdm import tqdm


def add_sex_feature(frequency_df):
    gender_dir = os.path.join(rootDirPath, "data", dataset)
    gender_data = pd.read_csv(f"{gender_dir}/{dataset}_gender.csv", sep=";")[
        ["ID", "H/F"]
    ]
    gender_data["Sex"] = gender_data["H/F"].apply(lambda x: 0 if x == "H" else 1)
    print(gender_data)
    print(frequency_df)
    frequency_df = frequency_df.set_index("ID").join(
        gender_data[["ID", "Sex"]].set_index("ID")
    )
    return frequency_df


def create_csv_files(features, data_dir):
    # Concat and remove columns automatically added by OpenSmile
    all_features = pd.concat(features).reset_index()
    all_features = all_features.drop(columns=["start", "end"])
    # Get only the audio id from the full path of the file
    all_features["ID"] = (
        all_features["file"].str.split("/").str[-1].str.split(".").str[0]
    )
    all_features = all_features.drop(columns=["file"])
    # Create right path for saving csv file
    feature_dir = os.path.join(data_dir, "features")
    csv_dir = os.path.join(feature_dir, clip, "audio")
    os.makedirs(csv_dir, exist_ok=True)
    # Get a dict with feature names associated with categories
    categoryDict = createFeatureLists()
    for cat, feat in categoryDict.items():
        current_category_feats = pd.concat(
            [all_features["ID"], all_features[feat]], axis=1
        )
        # if cat == "Frequency":
        #     current_category_feats = add_sex_feature(current_category_feats)
        # keep_index = cat == "Frequency"
        current_category_feats.to_csv(
            os.path.join(csv_dir, f"{cat.title()}.csv"), index=False
        )


def createFeatureLists():
    """
    Function creating the dictionary of the feature names of the category based
    on a json file containing them.

    Returns:
    categoryDict (dict): dictionary with the keys of category names and values
    as lists of feature names.
    """
    filepath = os.path.join(rootDirPath, "preprocess", "feature_categories.json")
    with open(filepath, "r") as cat:
        categoryDict = json.loads(cat.read())
    return categoryDict


def audioProcess():
    """
    Main function that reads .wav files from the directory with the audios and
    executes feature extraction. Audio directory should contain subfolders:
            full, beg, mid and end.
    Audio files are named with the ID in the dataset.

    Parameters:
    rootDirPATH: str: root directory
    dataset: name of the dataset

    Returns: features (DataFrame): array of N x nb_cat_feat dimensions where:
            n -- nb of data samples
            nb_cat_feat -- nb of features in the category
            category -- prosody, voice_quality, warmth, likability, confidence
    """
    # Find the audio paths based on the list of clips for the analysis
    data_dir = os.path.join(rootDirPath, "data", dataset)
    audio_dir = os.path.join(data_dir, "wav")
    audio_paths = glob.glob(f"{audio_dir}/{clip}/*.wav")
    # Initialize OpenSmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = []
    # Extract features with opensmile for every file and append to the list
    for audio in tqdm(audio_paths):
        features.append(smile.process_file(audio))
    create_csv_files(features, data_dir)


if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    from config_reader import read_config

    config = read_config()

    # Use the configuration values
    rootDirPath = config["rootDirPath"]
    dataset = config["dataset"]
    clip = config["clip"]

    # Example usage in your script
    print(f"Root Directory Path: {rootDirPath}")
    print(f"Dataset: {dataset}")
    print(f"Clip: {clip}")

    sys.path.append(rootDirPath)
    audioProcess()
