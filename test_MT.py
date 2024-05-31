from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess.FeatureSelection import z_test
import json
from feedback.SHAP import SVMInterpret
import numpy as np
import scipy.stats as st
from feedback.feedback_generator import feedback
from feedback.SHAP import grouped_shap
from joblib import dump,load

global dataset
dataset = 'MT'

global rate_type
rate_type = 'persuasiveness'

with open('./categories.json','r') as f:
    categories = json.load(f)



# print(categories)
audio_category = categories['audio_category']
# visual_category = categories['visual_category']
filler_category = categories['filler_category']
text_linking_rate_category = categories['text_linking_rate_category']
text_synonyms_rate_category = categories['text_synonyms_rate_category']

# text_embedding_category = categories['text_embedding_category']
text_div_category = categories['text_div_category']

text_dens_category = categories['text_dens_category']
text_disc_category = categories['text_disc_category']
text_ref_category = categories['text_ref_category']


# 'text_dens_category': text_dens_category,
#                 'text_disc_category': text_disc_category,
#                 'text_ref_category': text_ref_category

group_by_category = {**text_linking_rate_category, **text_synonyms_rate_category, **text_div_category, **text_dens_category, **text_disc_category, **text_ref_category}

# group_by_category = {**text_rate_category, **text_embedding_category, **text_div_category}
# group_by_category = {**audio_category, **filler_category, **visual_category}


#
# audio = pd.read_csv('./data/acoustic.csv', index_col=0)
# visual = pd.read_csv('./data/visual.csv', index_col=0)
# filler = pd.read_csv('./data/filler.csv', index_col=0)
text_linking_rate =  pd.read_csv('./data/' + dataset + '/text_linking_rate.csv', index_col=0)
text_synonym_rate =  pd.read_csv('./data/' + dataset + '/text_synonyms_rate.csv', index_col=0)

# text_embeddings =  pd.read_csv('./data/' + dataset + '/text_embeddings.csv', index_col=0)
text_div =  pd.read_csv('./data/' + dataset + '/text_diversity.csv', index_col=0)
text_dens =  pd.read_csv('./data/' + dataset + '/text_density.csv', index_col=0)
text_disc =  pd.read_csv('./data/' + dataset + '/text_discource.csv', index_col=0)
text_ref =  pd.read_csv('./data/' + dataset + '/text_reference.csv', index_col=0)


Y = pd.read_csv('./data/' + dataset + '/' + rate_type + 'Label.csv', index_col=0)

X = text_linking_rate.join(text_synonym_rate, how = 'left').join(text_div, how = 'left').join(text_dens, how = 'left').join(text_disc, how = 'left').join(text_ref, how = 'left')



print(X)

# audio = z_test(audio,Y)
# visual = z_test(visual,Y)
#print(text[2:])

# print(text_embeddings)

# text_embeddings = z_test(text_embeddings, Y)

# print(text_embeddings)
#
# print(text_rate.columns)
# print(text_embeddings.columns)

#
# print('audio',len(audio.columns))
# print('visual',len(visual.columns))


# X = audio.join(visual).join(filler)
# X = pd.read_csv('./demo/background.csv', index_col=0)

# print(np.shape(X))
# print(np.shape(Y))

# X = X.join(text_rate, how = 'left').join(text_embeddings, how = 'left')
# X = text_rate.join(text_embeddings, how = 'left').join(text_div, how = 'left')




# print(text)
# print(X.columns)
print(np.shape(X))
print(np.shape(Y))


target = X.loc[['ANT01']]
print(target)
feature_name = list(X.columns)


Y = Y.loc[(Y.index).isin(X.index)]
X = X.loc[(X.index).isin(Y.index)]
print(np.shape(X))
print(np.shape(Y))

# print(feature_name)

from Models.ML_Model import definedSVM, SupportVectorMachine
test_mean = []

def calculate_ci(data):
    mean = np.mean(data)
    ci = st.norm.interval(alpha=0.95, loc=mean, scale=st.sem(data))
    return mean, ci

#
# for i in range(0, 300):
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i, stratify=Y)
#     #rf_param, rf_train, rf_test, svm = SupportVectorMachine(X_train, y_train, X_test, y_test)
#     #print('best_param', rf_param)
#     #print('best_train_score', rf_train)
#     rf_test, svm = definedSVM(X_train, y_train, X_test, y_test)
#     test_mean.append(rf_test)
#     # print('test_score', rf_test)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
#rf_param, rf_train, rf_test, svm = SupportVectorMachine(X_train, y_train, X_test, y_test)
#print('best_param', rf_param)
#print('best_train_score', rf_train)
rf_test, svm = definedSVM(X_train, y_train, X_test, y_test)
# test_mean.append(rf_test)
# print('test_score', rf_test)

# mean, ci = calculate_ci(test_mean)


# txtForSave = open('./results/' + dataset + '/' + rate_type + '/F1_score.txt', 'w')
# txtForSave.write("mean: {}".format(mean) + " confidence interval: {}".format(ci) + "\n")
# txtForSave.close()

X.to_csv('./demo/' + dataset + '/' + rate_type + '/background.csv')
X_test.to_csv('./demo/' + dataset + '/' + rate_type + '/target.csv')
dump(svm,'./demo/' + dataset + '/' + rate_type + '/svm.joblib')


# print(X_test)
# print(mean, ci)
# print(y_test)
# print(svm.predict(X_test))
# print(svm.predict_proba(X_test))

feedback_generator = feedback('./results/' + dataset + '/' + rate_type + '/')
# group_shap = grouped_shap(svm,X_train,target,group_by_category,feature_name)
group_shap = grouped_shap(svm,X_train,X_test,group_by_category,feature_name)

feedback_generator.ABS_SHAP(group_shap)
SVMInterpret(svm, X_train, group_by_category,feature_name)

