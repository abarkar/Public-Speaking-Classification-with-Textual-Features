import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import sys
import glob


sys.path.append('/Users/goddessoffailures/Desktop/Mock-Up_sample/Mock-Up_sample/REVITALISE')


def tableToText(file):
    # with open(file,  encoding='latin-1') as f:
    #     for line in f.readlines():
    #         print(line[0])
    #
    # print("hello, blyat\n")


    data = pd.read_csv(file, sep='delimiter', header=None, encoding='latin-1')
    # print(data)
    data = np.array(data)
    text = []
    for row in data:
        # print(row)
        text.extend(word_tokenize(row[0]))
    # print(text)
    return text




# preprocess raw transcript and split it into sentences
def TranscriptProcess():
    # transcripts = os.listdir('./transcripts')
    # search all files inside a specific folder
    # *.* means file name with any extension
    dir_path = r'./transcripts/*.csv*'
    transcripts = glob.glob(dir_path)
    for file in transcripts:
        text = tableToText(file)
        txtForSave = open('./new_trans/' + text[0] + '.txt', 'w')
        for item in text[2:]:
            txtForSave.write(item + " ")
        txtForSave.close()
    #
    # print(transcripts)



if __name__ == "__main__":
    TranscriptProcess()


