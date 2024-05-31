import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Download MT180 ratings
MT_labels = pd.read_csv('./MT_labels_rms.csv', sep = ";", index_col = 0 , dtype = 'string')
columns = MT_labels.columns
index= MT_labels.index

#Converting rates of [confidence_rms  persuasiveness_rms  engagement_rms  global_rms ] to float64
cutted = np.array(MT_labels[MT_labels.columns[0:]])
cutted = [[str.replace(",", ".") for str in row[:-2]] for row in cutted]
cutted = pd.DataFrame(cutted, index = index, columns = columns[0:-2], dtype = np.float64)
columns_cutted = cutted.columns
MT_labels[columns_cutted] = cutted[columns_cutted]
MT_labels = MT_labels[MT_labels['video'] == 'full']
# print(MT_labels)

# Download POM ratings
POM_labels = pd.read_csv('./POM_labels.csv', sep = ",", index_col = 0 , skiprows = 1, dtype = np.float64)
columns = POM_labels.columns
index= POM_labels.index

print(POM_labels)

print(columns)

print(index)




fig, axes = plt.subplots(nrows = 6, ncols =3, figsize = (8, 6))
colors = ["#e24a33", "#348abd", "#988ed5", "#777777", 'red', 'pink', 'yellow', "#e24a33", "#348abd", "#988ed5", "#777777", 'red', 'pink', 'yellow',  "#e24a33", "#348abd", "#988ed5", "#777777",] # whatever the colors may be but it should be different for each histogram.

for index, score in enumerate(columns):
    ax = axes.flatten()[index]
    ax.hist(POM_labels[score], color = colors[index], label = score)
    ax.legend(loc = "best")
    ax.grid(visible = True)
# plt.suptitle("Histograms of DS characteristics", size = 20)
plt.show()
