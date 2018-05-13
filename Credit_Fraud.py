'''
The goal of this project is to predict fradulent credit card transations. The
predictors have been anonymized by PCA.

There is an initial analysis in R. This script picks up where the R version left
off, as computational speed was a limiting factor. It repeats the same sort of
analyses that I did in R, but with slightly less explanation.
'''

import os
import random
random.seed(123456)
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from Credit_Fraud_functions import *

credit = pd.read_csv('Data/creditcard.csv')


# Time represents seconds since 12AM (time 0), spanning over a 48 hour period.
# Create a new variable that converts these time diffs to hour of the day (as floats)
mask = credit['Time'] > 86400
credit['time_of_day'] = np.where(credit['Time'] > 86400,
                                 (credit['Time'] - 86400) / 3600,
                                 credit['Time'] / 3600)
credit = credit.drop('Time', axis=1)


##
### Data Partitioning
##

# Divide into train, dev, and test sets
from sklearn.model_selection import train_test_split

credit_train, temp = train_test_split(credit, test_size=0.3)
credit_dev, credit_test = train_test_split(temp, test_size=0.5)

# Split response (y) variable from predictors (X)
credit_train_x = credit_train.drop('Class', 1)
credit_train_y = credit_train['Class']

credit_dev_x = credit_dev.drop('Class', 1)
credit_dev_y = credit_dev['Class']

credit_test_x = credit_test.drop('Class', 1)
credit_test_y = credit_test['Class']


##
### Sampling techniques for imbalanced classes
##

## Undersampling

# Generate undersampled data
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0, replacement=False)
credit_train_x_undersampled, credit_train_y_undersampled = rus.fit_sample(credit_train_x, credit_train_y)
print(credit_train_x_undersampled.shape)

# Perform a L1 penalized logistics regression grid search for lambda
param_grid = [{'C': [0.001, 0.0015, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'penalty': ['l1']}]
scores = ['f1']
# This function is a simple wrapper around sk-learn functions (Credit_Fraud_functions.py)
# It performs a grid search using provided parameters and prints the results.
# Prints training metrics for each combination of parameters and tests the
# best training model against the dev set.
clf_log_under = auto_grid_search_clf(x_train=credit_train_x_undersampled,
                                     y_train=credit_train_y_undersampled,
                                     parameter_grid=param_grid,
                                     x_test=credit_dev_x,
                                     y_test=credit_dev_y,
                                     metrics=scores)

## Plot Performance vs. lambda
plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                 clf_log_under.cv_results_['mean_test_score'],
                 error=clf_log_under.cv_results_['std_test_score'],
                 xlab = 'log10(lambda)')


## Plot of coefficients for variable importance

# Plot the coefficients
variable_names = pd.Series(credit_train_x.columns.values)
plot_varimp_logistic(clf_log_under, variable_names)


## Plot precision vs. recall curve
plot_precision_recall(clf_log_under.decision_function(credit_dev_x), credit_dev_y)


## Naive Oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)

# For computational speed (I'm running this on a laptop), I'm going to cut-down
# the size of the training set substantially.
credit_train_cut = credit_train.sample(n=40000)
credit_train_cut_x = credit_train_cut.drop('Class', 1)
credit_train_cut_y = credit_train_cut['Class']

credit_train_x_oversampled, credit_train_y_oversampled = \
    ros.fit_sample(credit_train_cut_x, credit_train_cut_y)

print(credit_train_x_oversampled.shape)

# Perform a L1 penalized logistics regression grid search for lambda
# Keep the same parameters as the undersampled model.
clf_log_over = auto_grid_search_clf(x_train=credit_train_x_oversampled,
                                     y_train=credit_train_y_oversampled,
                                     parameter_grid=param_grid,
                                     x_test=credit_dev_x,
                                     y_test=credit_dev_y,
                                     metrics=scores)

## Plot Performance vs. lambda
plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                 clf_log_over.cv_results_['mean_test_score'],
                 error=clf_log_over.cv_results_['std_test_score'],
                 xlab = 'log10(lambda)')

## Plot variable importance
plot_varimp_logistic(clf_log_over, variable_names)

## Plot precision vs. recall curve
plot_precision_recall(clf_log_over.decision_function(credit_dev_x), credit_dev_y)


## SMOTE Oversampling
from imblearn.over_sampling import SMOTE
credit_train_x_smote, credit_train_y_smote = \
    SMOTE(kind='borderline2').fit_sample(credit_train_cut_x, credit_train_cut_y)
print(credit_train_x_smote.shape)


# Perform a L1 penalized logistics regression grid search for lambda
# Keep the same parameters as the undersampled model.
clf_log_smote = auto_grid_search_clf(x_train=credit_train_x_smote,
                                     y_train=credit_train_y_smote,
                                     parameter_grid=param_grid,
                                     x_test=credit_dev_x,
                                     y_test=credit_dev_y,
                                     metrics=scores)

## Plot Performance vs. lambda
plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                 clf_log_smote.cv_results_['mean_test_score'],
                 error=clf_log_smote.cv_results_['std_test_score'],
                 xlab = 'log10(lambda)')

## Plot variable importance
plot_varimp_logistic(clf_log_smote, variable_names)

## Plot precision vs. recall curve
plot_precision_recall(clf_log_smote.decision_function(credit_dev_x), credit_dev_y)

