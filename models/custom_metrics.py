import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# Define metric functions
def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def mae_metric(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mse_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def rmse_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def r2_metric(y_true, y_pred):
    return r2_score(y_true, y_pred)

def mape_metric(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def medae_metric(y_true, y_pred):
    return median_absolute_error(y_true, y_pred)
