# SHAP related methods

import shap
from itertools import repeat, chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/goddessoffailures/Desktop/Mock-Up_sample/Mock-Up_sample/REVITALISE')


global dataset
dataset = 'MT'

global rate_type
rate_type = 'persuasiveness'

revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))

# calculate importance of each category
def SVMInterpret(model, X, groups, features_name):
    group_shap = grouped_shap(model, X, X, groups, features_name)
    txtForSave = open('./results/' + dataset + '/' + rate_type + '/mean_interpret.txt', 'w')

    for col_0 in group_shap.columns:
        tmp = group_shap[col_0].values
        tmp = np.abs(tmp)
        txtForSave.write(col_0 + ": {}".format(tmp.mean()) + '\n')

        # print(col_0 + ": {}".format(tmp.mean()))
    txtForSave.close()


def plot_shap_by_feature(shap_values, target):
    print("Plot contribution")
    shap.summary_plot(shap_values, target, max_display=10, auto_size_plot=True)
    plt.savefig('./results/' + dataset + '/' + rate_type + 'contributionSHAP.png')


# calculate Shapley value of each group
def grouped_shap(model, background, target, groups, features_name):
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(target)
    # shap_values_all = explainer.shap_values(background)

    #matrix of SHAP values (# samples x # features). Each row sums to the difference between the model
    # output for that sample and the expected value of the model output (which is stored as expected_value attribute of the explainer).

    print("there is shpa w.r.t. features")
    shap.summary_plot(shap_values[1], target, max_display=10, auto_size_plot=True)
    print("finished")

    # plot_shap_by_feature(shap_values, target)

    shap_0 = shap_values[1]
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_0, columns=pd.Index(features_name, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T


    print(shap_grouped)

    return shap_grouped
