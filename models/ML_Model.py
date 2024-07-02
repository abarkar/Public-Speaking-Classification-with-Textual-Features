"""
Implementation of classifiers class. 

TODO: Add regression models.

Classifier class has GridSearch and Defined functions that create eather 
search for the parameters or fit and test the model with predefined parameters. 

Use GridSearch for the research goals and Defined models for the actual 
predictions on the new data. 

Attributes:
self.clf : model name which is passed to the model when creating the class object.
self.scoring : scoring method used in the grid search (accuracy/f1 for classification).

Return:
Grid Search: best_param, train_score, test_score, {self.clf}_best
Defined: accuracy, f1

best_param (dict): Parameter setting that gave the best results on the hold out data.
train_score (float): Mean cross-validated score of the best_estimator.
test_score (float): Score of best_estimator on the test set.
{self.clf}_best (estimator): Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss if specified) on the left out data. 

@author: Alisa Barkar, alisa.george.barkar@gmail.com
"""
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
import models.custom_metrics as cm
import numpy as np


class classificator:
    def __init__(self, clf_type):
        self.clf = clf_type
        self.scoring = "accuracy"
        self.target_metric=cm.f1_metric

    def GridSearch(self, X_train, Y_train, X_test, Y_test):
        if self.clf == "SVM":
            return self.SupportVectorMachine(X_train, Y_train, X_test, Y_test)
        elif self.clf == "RFC":
            return self.RandomForest(X_train, Y_train, X_test, Y_test)
        elif self.clf == "LR":
            return self.LogRegression(X_train, Y_train, X_test, Y_test)
        elif self.clf == "NB":
            return self.NBayes(X_train, Y_train, X_test, Y_test)
        elif self.clf == "KN":
            return self.KNeigbors(X_train, Y_train, X_test, Y_test)
        elif self.clf == "LIN":
            self.target_metric=cm.rmse_metric
            return self.LinReg(X_train, Y_train, X_test, Y_test)
        elif self.clf == "RID":
            self.target_metric=cm.rmse_metric
            return self.RidgeReg(X_train, Y_train, X_test, Y_test)
        elif self.clf == "LAS":
            self.target_metric=cm.rmse_metric
            return self.LassoReg(X_train, Y_train, X_test, Y_test)
        elif self.clf == "eNET":
            self.target_metric=cm.rmse_metric
            return self.ElasticNetReg(X_train, Y_train, X_test, Y_test)
        elif self.clf == "RFR":
            self.target_metric=cm.rmse_metric
            return self.RandomForestReg(X_train, Y_train, X_test, Y_test)
        else:
            print("This classificator is not implemented\n")
            return None
    def Defined(self, X_train, Y_train, X_test, Y_test, best_param):
        if self.clf == "SVM":
            return self.definedSVM(X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "RFC":
            return self.definedRFC(X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "LR":
            return self.definedLR( X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "NB":
            return self.definedNB( X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "KN":
            return self.definedKN(X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "LIN":
            return self.definedLinReg(X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "RID":
            return self.definedRidgeReg(X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "LAS":
            return self.definedLassoReg(X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "eNET":
            return self.definedElasticNetReg(X_train, Y_train, X_test, Y_test, best_param)
        elif self.clf == "RFR":
            return self.definedRandomForgestReg(X_train, Y_train, X_test, Y_test, best_param)
        else:
            print("This classificator is not implemented\n")
            return None

    def SupportVectorMachine(self, X_train, Y_train, X_test, Y_test):
        svc_param_grid = {'kernel': ['rbf', 'linear'],
                          'probability':[True],
                          'gamma': [0.001, 0.01, 0.1, 1,'auto'],
                          'C': [1, 10, 20]}
        SVMC = svm.SVC()
        gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=10, scoring=self.scoring, n_jobs=-1, verbose=1, refit=True)
        gsSVMC.fit(X_train, Y_train.values.ravel())
        SVMC_best = gsSVMC.best_estimator_
        best_param = gsSVMC.best_params_
        train_score = gsSVMC.best_score_

        test_score = SVMC_best.score(X_test, Y_test)
        return best_param, train_score, test_score, SVMC_best

    def RandomForest(self, X_train, Y_train, X_test, Y_test):
        RFC = RandomForestClassifier()
        ## Search grid for optimal parameters
        rf_param_grid = {"max_depth": [None],
                         "max_features": [10,20, 'sqrt'],
                         "min_samples_split": [2, 3, 10],
                         "min_samples_leaf": [1, 3, 10],
                         "bootstrap": [True],
                         "n_estimators": [300,400],
                         "criterion": ["gini"]}
        gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=5, scoring=self.scoring, n_jobs=-1, verbose=1)
        gsRFC.fit(X_train, Y_train.values.ravel())
        RFC_best = gsRFC.best_estimator_
        best_param = gsRFC.best_params_
        train_score = gsRFC.best_score_
        test_score = RFC_best.score(X_test, Y_test)
        importance = RFC_best.feature_importances_
        return best_param, train_score, test_score, RFC_best

    def LogRegression(self, X_train, Y_train, X_test, Y_test):
        LR = LogisticRegression()
        # ## Search grid for optimal parameters
        # lr_param_grid = {"penalty": ['l1', 'l2', 'elasticnet', 'none'],
        #                  "C": [0.001, 0.01, 0.1, 1., 10, 20],
        #                  "fit_intercept": [True, False],
        #                  "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        #                  "multi_class": ['ovr'],
        #                  "l1_ratio": [0., 0.001, 0.1, 0.5, 0.7, 0.9, 1.]}

        lr_param_grid = {"penalty": ['l1', 'l2'],
                         "C": [0.001, 0.01, 0.1, 1., 10, 20],
                         "multi_class": ['ovr'],
                         "solver": ['liblinear'],
                         "max_iter": [1000]}
        gsLR = GridSearchCV(LR, param_grid=lr_param_grid, cv=5, scoring=self.scoring, n_jobs=-1, verbose=1)
        gsLR.fit(X_train, Y_train.values.ravel())
        LR_best = gsLR.best_estimator_
        best_param = gsLR.best_params_
        train_score = gsLR.best_score_
        test_score = LR_best.score(X_test, Y_test)
        return best_param, train_score, test_score, LR_best

    def NBayes(self, X_train, Y_train, X_test, Y_test):
        NB = GaussianNB()
        nb_param_grid = {}
        gsNB = GridSearchCV(NB, param_grid=nb_param_grid, cv=5, scoring=self.scoring, n_jobs=-1, verbose=1)
        gsNB.fit(X_train, Y_train.values.ravel())
        NB_best = gsNB.best_estimator_
        best_param = gsNB.best_params_
        train_score = gsNB.best_score_
        test_score = NB_best.score(X_test, Y_test)
        return best_param, train_score, test_score, NB_best

    def KNeigbors(self, X_train, Y_train, X_test, Y_test):
        KN = KNeighborsClassifier()
        kn_param_grid = {'n_neighbors': [3, 5, 7],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        gsKN = GridSearchCV(KN, param_grid=kn_param_grid, cv=5, scoring=self.scoring, n_jobs=-1, verbose=1)
        gsKN.fit(X_train, Y_train.values.ravel())
        KN_best = gsKN.best_estimator_
        best_param = gsKN.best_params_
        train_score = gsKN.best_score_
        test_score = KN_best.score(X_test, Y_test)
        return best_param, train_score, test_score, KN_best
    


    def LinReg(self, X_train, Y_train, X_test, Y_test):
        model = LinearRegression()
        model.fit(X_train, Y_train)
        best_param={}
        best_param['coefficients']=model.coef_
        best_param['intercept']=model.intercept_
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)
        train_score = self.target_metric(Y_train, Y_pred_train)
        test_score = self.target_metric(Y_test, Y_pred_test) 
        return best_param, train_score, test_score, model


    def RidgeReg(self, X_train, Y_train, X_test, Y_test):
        model = Ridge()
        # Define the range of alpha values to try
        param_grid ={'alpha':[0.1, 1.0, 10.0]}
        # Perform randomized search cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
        # Fit Grid Search
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters
        best_model = grid_search.best_estimator_
        best_param = grid_search.best_params_ 
        # print("best_param of Ridge regression in GS:", best_param)
        train_score = grid_search.best_score_
        test_score = grid_search.score(X_test, Y_test)
        return best_param, train_score, test_score, best_model



    def LassoReg(self, X_train, Y_train, X_test, Y_test):
        model = Lasso()
        # Define the range of alpha values to try
        param_grid ={'alpha':[0.1, 1.0, 10.0]}
        # Perform grid search cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters
        best_model = grid_search.best_estimator_
        best_param = grid_search.best_params_
        train_score = grid_search.best_score_
        test_score = grid_search.score(X_test, Y_test)
        return best_param, train_score, test_score, best_model


    def ElasticNetReg(self, X_train, Y_train, X_test, Y_test):
        model = ElasticNet()
        # Define the range of alpha and l1_ratio values to try
        param_grid={'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}
        # Perform grid search cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters
        best_model = grid_search.best_estimator_
        best_param = grid_search.best_params_
        # print(best_param)
        train_score = grid_search.best_score_
        test_score = grid_search.score(X_test, Y_test)
        return best_param, train_score, test_score, best_model
        
        
    def RandomForestReg(self, X_train, Y_train, X_test, Y_test):
        model = RandomForestRegressor()
        # Define the hyperparameters to tune
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        # Perform grid search cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters
        best_model = grid_search.best_estimator_
        best_param = grid_search.best_params_
        train_score = grid_search.best_score_
        test_score = grid_search.score(X_test, Y_test)
        return best_param, train_score, test_score, best_model
    
    def definedLinReg(self, X_train, Y_train, X_test, Y_test, best_param):
        # Create a new LinearRegression instance
        defined_model = LinearRegression()
        # Manually set the coefficients and intercept
        defined_model.coef_ = best_param['coefficients']
        defined_model.intercept_ = best_param["intercept"]
        # Fit defined model
        defined_model.fit(X_train,Y_train.values.ravel())
        Y_pred = defined_model.predict(X_test)
        # Metrics for defined model
        mae = cm.mae_metric(Y_test, Y_pred)
        rmse = cm.rmse_metric(Y_test, Y_pred)
        return mae, rmse
    
    def definedLassoReg(self, X_train, Y_train, X_test, Y_test, best_param):
        # Create a new LinearRegression instance
        defined_model = Lasso(alpha=best_param['alpha'])
        # Manually set the coefficients and intercept
        # defined_model.coef_ = best_param['alpha']
        # defined_model.intercept_ = best_param["intercept"]
        # Fit defined model
        defined_model.fit(X_train,Y_train.values.ravel())
        Y_pred = defined_model.predict(X_test)
        # Metrics for defined model
        mae = cm.mae_metric(Y_test, Y_pred)
        rmse = cm.rmse_metric(Y_test, Y_pred)
        return mae, rmse
    
    def definedElasticNetReg(self, X_train, Y_train, X_test, Y_test, best_param):
        # Create a new LinearRegression instance
        defined_model = ElasticNet(alpha=best_param['alpha'], l1_ratio=best_param['l1_ratio'])
        # Manually set the coefficients and intercept
        # defined_model.alpha = best_param['alpha']
        # defined_model.l1_ratio = best_param['l1_ratio']
        # print(defined_model.coef_ )
        # defined_model.coef_ = best_param['alpha']
        # Fit defined model
        defined_model.fit(X_train,Y_train.values.ravel())
        Y_pred = defined_model.predict(X_test)
        # Metrics for defined model
        mae = cm.mae_metric(Y_test, Y_pred)
        rmse = cm.rmse_metric(Y_test, Y_pred)
        return mae, rmse
    
    def definedRandomForgestReg(self, X_train, Y_train, X_test, Y_test, best_param):
        # Create a new LinearRegression instance
        defined_model = ElasticNet()
        # Manually set the coefficients and intercept
        defined_model.n_estimators = best_param['n_estimators']
        defined_model.max_depth = best_param['max_depth']
        defined_model.min_samples_split = best_param['min_samples_split']
        defined_model.min_samples_leaf = best_param['min_samples_leaf']
        # Fit defined model
        defined_model.fit(X_train,Y_train.values.ravel())
        Y_pred = defined_model.predict(X_test)
        # Metrics for defined model
        mae = cm.mae_metric(Y_test, Y_pred)
        rmse = cm.rmse_metric(Y_test, Y_pred)
        return mae, rmse


    def definedRidgeReg(self, X_train, Y_train, X_test, Y_test, best_param):
        # Create a new LinearRegression instance
        defined_model = Ridge(alpha=best_param['alpha'])
        # Manually set the coefficients and intercept
        defined_model.coef_ = best_param['alpha']
        # defined_model.intercept_ = best_param["intercept"]
        # Fit defined model
        defined_model.fit(X_train,Y_train.values.ravel())
        Y_pred = defined_model.predict(X_test)
        # Metrics for defined model
        mae = cm.mae_metric(Y_test, Y_pred)
        rmse = cm.rmse_metric(Y_test, Y_pred)
        # Now target m
        return rmse
        

    def definedKN(self, X_train, Y_train, X_test, Y_test, best_param):
        clf = KNeighborsClassifier(n_neighbors = best_param["n_neighbors"])
        # Train the classifier
        clf.fit(X_train, Y_train.values.ravel())
        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred, zero_division=0.)
        return accuracy, f1
    

    def definedNB(self, X_train, Y_train, X_test, Y_test, best_param):
        clf = GaussianNB()
        # Train the classifier
        clf.fit(X_train, Y_train.values.ravel())
        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred, zero_division=0.)
        return accuracy, f1

    def definedLR(self, X_train, Y_train, X_test, Y_test, best_param):
        clf = LogisticRegression(max_iter = best_param["max_iter"], solver = best_param["solver"], penalty = best_param["penalty"],C = best_param["C"], multi_class = best_param["multi_class"])
        # Train the classifier
        clf.fit(X_train, Y_train.values.ravel())
        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        # Calculate the accuracy of the classifier

        print("------------- F1 average test ----------")
        print("Predicted: \n", y_pred , "\n")
        print("Real: \n", Y_test, "\n")
        accuracy = balanced_accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred, zero_division=0.)
        return accuracy, f1

    def definedSVM(self, X_train, Y_train, X_test, Y_test, best_param):
        SVM = svm.SVC(kernel=best_param["kernel"], probability=best_param["probability"], gamma=best_param["gamma"], C=best_param["C"])
        SVM.fit(X_train,Y_train.values.ravel())
        y_pred = SVM.predict(X_test)
        # print("------------- F1 average test ----------")
        # print("Predicted: \n", y_pred , "\n")
        # print("Real: \n", Y_test, "\n")
        f1 = f1_score(Y_test, y_pred, zero_division=0.)
        acc = accuracy_score(Y_test, y_pred)
        return acc, f1

    def definedRFC(self, X_train, Y_train, X_test, Y_test, best_param):
        clf = RandomForestClassifier(max_features = best_param["max_features"], min_samples_split = best_param["min_samples_split"], min_samples_leaf = best_param["min_samples_leaf"], bootstrap = best_param["bootstrap"], n_estimators = best_param["n_estimators"], criterion = best_param["criterion"])
        # Train the classifier
        clf.fit(X_train, Y_train.values.ravel())
        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        # Calculate the accuracy of the classifier
        # accuracy = balanced_accuracy_score(Y_test, y_pred)
        # print("------------- F1 average test ----------")
        # print("Predicted: \n", y_pred, "\n")
        # print("Real: \n", Y_test, "\n")

        accuracy = accuracy_score(Y_test, y_pred)

        f1 = f1_score(Y_test, y_pred, zero_division=0.)
        return accuracy, f1

