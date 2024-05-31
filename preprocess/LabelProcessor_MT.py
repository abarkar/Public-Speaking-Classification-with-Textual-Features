# Select high and low confidence videos from original dataset

from mmsdk import mmdatasdk

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import sys

sys.path.append('/Users/goddessoffailures/Desktop/Mock-Up_sample/Mock-Up_sample/REVITALISE')

rate_type = 'persuasiveness' + '_rms'


def extractConfidenceLabels():

    # To work with raw data from MT_Labels.csv:
    # MT_labels = pd.read_csv('../data/MT/MT_Labels.csv', sep=";", index_col='Input.name', dtype='string', encoding='latin-1')
    # MT_labels = MT_labels[MT_labels['clip'] == 'full']
    # MT_labels = MT_labels[['Answer.Competence', 'Answer.Engagement', 'Answer.Global', 'Answer.Persuasiveness']]
    # index = MT_labels.index

    # To work with mean built in the file MT_Labels_rms.csv

    # Download MT180 ratings
    MT_labels = pd.read_csv('../data/MT/MT_labels_rms.csv', sep=";", index_col=-1, dtype='string')
    MT_labels = MT_labels[MT_labels['video'] == 'full']
    columns = MT_labels.columns
    index = MT_labels.index

    # Converting rates of [confidence_rms  persuasiveness_rms  engagement_rms  global_rms ] to float64
    cutted = np.array(MT_labels[MT_labels.columns[1:-1]])
    cutted = [[str.replace(",", ".") for str in row[:]] for row in cutted]
    MT_labels = pd.DataFrame(cutted, index=index, columns=columns[1:-1], dtype=np.float64)


    data = pd.DataFrame()

    median = MT_labels.median()
    print("median: ", median)

    for index, row in MT_labels.iterrows():
        # print(index, row)

        if row[rate_type] >= median[rate_type]:
            data = data.append(pd.DataFrame([1], columns=['label'], index=[index]))
        if row[rate_type] < median[rate_type]:
            data = data.append(pd.DataFrame([0], columns=['label'], index=[index]))

    # print(data)
    class_1 = len(data.loc[data['label'] == 1])
    class_0 = len(data.loc[data['label'] == 0])
    print("class_1: ", class_1, " class_0: ", class_0)

    # print(data['label'].value_counts())
    # # print(len(pers_keys))
    data.to_csv('../data/MT/globalLabel.csv')

if __name__ == "__main__":
    extractConfidenceLabels()