import os
import sys
import pandas as pd
from config_reader import read_config


def separateGenders():
    gender_dir = os.path.join(rootDirPath, "data", dataset)
    gender_data = pd.read_csv(f"{gender_dir}/{dataset}_gender.csv", sep=";")[
        ["ID", "H/F"]
    ]
    for dim in dimension:
        labels_dir = os.path.join(rootDirPath, "data", dataset, "labels", clip)
        labels = pd.read_csv(f"{labels_dir}/{dim}Label.csv")
        labels_and_genders = labels.set_index("ID").join(gender_data.set_index("ID"))
        women_labels = labels_and_genders.query("`H/F` == 'F'")[["label"]]
        men_labels = labels_and_genders.query("`H/F` == 'H'")[["label"]]
        women_labels.to_csv(f"{labels_dir}/{dim}Label_women.csv")
        men_labels.to_csv(f"{labels_dir}/{dim}Label_men.csv")
        print(f"Men labels: {len(men_labels.query('`label` == 0'))} are 0, {len(men_labels.query('`label` == 1'))} are 1.")
        print(f"Women labels: {len(women_labels.query('`label` == 0'))} are 0, {len(women_labels.query('`label` == 1'))} are 1.")


if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    from config_reader import read_config

    config = read_config()

    # Use the configuration values
    rootDirPath = config["rootDirPath"]
    dataset = config["dataset"]
    dimension = config["dimension"]
    clip = config["clip"]

    print(f"Root Directory Path: {rootDirPath}")
    print(f"Dataset: {dataset}")
    print(f"Dimension: {dimension}")
    print(f"Clip: {clip}")

    sys.path.append(rootDirPath)
    separateGenders()
