'''
The goal of this project is to predict fraudulent credit card transations. The
predictors have been anonymized by PCA.

There is an initial analysis in R. This script picks up where the R version left
off, as computational speed was a limiting factor. It repeats the same sort of
analyses that I did in R, but with slightly less explanation.

For the sake of decision-making, this project will use the following implicit
weights:
Cost of a False Negative = 4
Cost of a False Positive = 1

We are willing to falsely flag 4 transactions as fraudulent in order to catch 1.
That is, we are willing to accept a Positive Predictive Value (PPV) as low as 20%.

This script fundamentally has the same code as the Jupyter Notebook, but it is
arranged so that the data can be imported into another script without running
the models
'''


rand_state = 123456
np.random.seed(rand_state)
from Credit_Fraud_functions import *
from sklearn.metrics import confusion_matrix

credit_1 = pd.read_csv('Data/creditcard_1.csv')
credit_2 = pd.read_csv('Data/creditcard_2.csv')
credit = credit_1.append(credit_2, ignore_index=True)
del credit_1, credit_2


# There is one observation with a missing entry, which will be excluded from analysis.
credit = credit.dropna()

# Create a new variable that converts these time diffs to hour of the day (as floats)
mask = credit['Time'] > 86400
credit['time_of_day'] = np.where(credit['Time'] > 86400,
                                 (credit['Time'] - 86400) / 3600,
                                 credit['Time'] / 3600)
credit = credit.drop('Time', axis=1)



##
### Data Partitioning
##

# Before partitioning the data, standardize the scale of the variables
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler(with_centering=True, with_scaling=True)

credit = pd.DataFrame(scaler.fit_transform(credit), columns=credit.columns)


# Divide into train, dev, and test sets
from sklearn.model_selection import train_test_split

credit_train, temp = train_test_split(credit, test_size=0.3, stratify=credit['Class'])
credit_dev, credit_test = train_test_split(temp, test_size=0.5, stratify=temp['Class'])

# Split response (y) variable from predictors (X)
credit_train_x = credit_train.drop('Class', 1)
credit_train_y = credit_train['Class']

credit_dev_x = credit_dev.drop('Class', 1)
credit_dev_y = credit_dev['Class']

credit_test_x = credit_test.drop('Class', 1)
credit_test_y = credit_test['Class']

variable_names = pd.Series(credit_train_x.columns.values)
##
### Sampling techniques for imbalanced classes
##



## Undersampling

# Generate undersampled data
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=rand_state, replacement=False)
credit_train_x_undersampled, credit_train_y_undersampled = rus.fit_sample(credit_train_x, credit_train_y)


## Naive Oversampling
from imblearn.over_sampling import RandomOverSampler

# For computational speed (I'm running this on a laptop), I'm going still going
# to do some undersampling. Keep all positive classes and cut down
temp = credit_train[credit_train['Class'] == 1]
temp2 = credit_train[credit_train['Class'] == 0]
credit_train_cut = temp2.sample(n = 40000).append(temp, ignore_index = True)

credit_train_cut_x = credit_train_cut.drop('Class', 1)
credit_train_cut_y = credit_train_cut['Class']

ros = RandomOverSampler(random_state=rand_state)
credit_train_x_oversampled, credit_train_y_oversampled = \
    ros.fit_sample(credit_train_cut_x, credit_train_cut_y)


## SMOTE Oversampling -- borderline2
# Borderline to uses samples 'in danger' to generate new observations
from imblearn.over_sampling import SMOTE

credit_train_x_smote, credit_train_y_smote = \
    SMOTE(kind='borderline2').fit_sample(credit_train_cut_x, credit_train_cut_y)



## ADASYN Oversampling
from imblearn.over_sampling import ADASYN

credit_train_x_adasyn, credit_train_y_adasyn = \
    ADASYN().fit_sample(credit_train_cut_x, credit_train_cut_y)


## SMOTE Oversampling -- SVM
# Uses an svm to generate new observations.
credit_train_x_smote_svm, credit_train_y_smote_svm = \
    SMOTE(kind='svm').fit_sample(credit_train_cut_x, credit_train_cut_y)



##
### Run Models
##

if __name__ == '__main__':
    ## Undersampling

    # Perform a L1 penalized logistics regression grid search for lambda
    param_grid = [{'C': [0.001, 0.0015, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                   'penalty': ['l1'],}]
    scores = ['f1_macro']
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

    ## Confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix(credit_dev_y, clf_log_under.predict(credit_dev_x))

    ## Plot Performance vs. lambda
    plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                     clf_log_under.cv_results_['mean_test_score'],
                     error=clf_log_under.cv_results_['std_test_score'],
                     xlab = 'log10(lambda)')


    ## Plot of coefficients for variable importance

    # Plot the coefficients

    plot_varimp(clf_log_under, variable_names)


    ## Plot precision vs. recall curve
    plot_precision_recall(clf_log_under.decision_function(credit_dev_x), credit_dev_y)



    ## Naive Oversampling

    # Perform a L1 penalized logistics regression grid search for lambda
    # Keep the same parameters as the undersampled model.
    clf_log_over = auto_grid_search_clf(x_train=credit_train_x_oversampled,
                                         y_train=credit_train_y_oversampled,
                                         parameter_grid=param_grid,
                                         x_test=credit_dev_x,
                                         y_test=credit_dev_y,
                                         metrics=scores)

    ## Confusion matrix
    confusion_matrix(credit_dev_y, clf_log_under.predict(credit_dev_x))

    ## Plot Performance vs. lambda
    plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                     clf_log_over.cv_results_['mean_test_score'],
                     error=clf_log_over.cv_results_['std_test_score'],
                     xlab = 'log10(lambda)')

    ## Plot variable importance
    plot_varimp(clf_log_over, variable_names)

    ## Plot precision vs. recall curve
    plot_precision_recall(clf_log_over.decision_function(credit_dev_x), credit_dev_y)



    ## SMOTE (Borderline2)

    # Perform a L1 penalized logistics regression grid search for lambda
    # Keep the same parameters as the undersampled model.
    clf_log_smote = auto_grid_search_clf(x_train=credit_train_x_smote,
                                         y_train=credit_train_y_smote,
                                         parameter_grid=param_grid,
                                         x_test=credit_dev_x,
                                         y_test=credit_dev_y,
                                         metrics=scores)

    ## Confusion matrix
    confusion_matrix(credit_dev_y, clf_log_under.predict(credit_dev_x))

    ## Plot Performance vs. lambda
    plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                     clf_log_smote.cv_results_['mean_test_score'],
                     error=clf_log_smote.cv_results_['std_test_score'],
                     xlab = 'log10(lambda)')

    ## Plot variable importance
    plot_varimp(clf_log_smote, variable_names)

    ## Plot precision vs. recall curve
    plot_precision_recall(clf_log_smote.decision_function(credit_dev_x), credit_dev_y)




    ## SMOTE SVM

    # Perform a L1 penalized logistics regression grid search for lambda
    # Keep the same parameters as the undersampled model.
    clf_log_smote_svm = auto_grid_search_clf(x_train=credit_train_x_smote_svm,
                                         y_train=credit_train_y_smote_svm,
                                         parameter_grid=param_grid,
                                         x_test=credit_dev_x,
                                         y_test=credit_dev_y,
                                         metrics=scores)

    ## Confusion matrix
    confusion_matrix(credit_dev_y, clf_log_under.predict(credit_dev_x))

    ## Plot Performance vs. lambda
    plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                     clf_log_smote_svm.cv_results_['mean_test_score'],
                     error=clf_log_smote_svm.cv_results_['std_test_score'],
                     xlab = 'log10(lambda)')

    ## Plot variable importance
    plot_varimp(clf_log_smote_svm, variable_names)

    ## Plot precision vs. recall curve
    plot_precision_recall(clf_log_smote_svm.decision_function(credit_dev_x), credit_dev_y)


    ## ADASYN Oversampling

    # Perform a L1 penalized logistics regression grid search for lambda
    # Keep the same parameters as the undersampled model.
    clf_log_adasyn = auto_grid_search_clf(x_train=credit_train_x_adasyn,
                                         y_train=credit_train_y_adasyn,
                                         parameter_grid=param_grid,
                                         x_test=credit_dev_x,
                                         y_test=credit_dev_y,
                                         metrics=scores)

    ## Confusion matrix
    confusion_matrix(credit_dev_y, clf_log_under.predict(credit_dev_x))

    ## Plot Performance vs. lambda
    plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                     clf_log_adasyn.cv_results_['mean_test_score'],
                     error=clf_log_adasyn.cv_results_['std_test_score'],
                     xlab = 'log10(lambda)')

    ## Plot variable importance
    plot_varimp(clf_log_adasyn, variable_names)

    ## Plot precision vs. recall curve
    plot_precision_recall(clf_log_adasyn.decision_function(credit_dev_x), credit_dev_y)




    ## No Sampling
    # As a baseline-- How do these more sophisticated sampling techniques compare to
    # just ignoring the imbalanced classes?

    # I'm still doing significant undersampling for computational reasons--
    # 40,000 negative cases

    # Perform a L1 penalized logistics regression grid search for lambda
    # Keep the same parameters as the undersampled model.
    param_grid = [{'C': [0.001, 0.0015, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                   'penalty': ['l1'],
                   'class_weight': [{0: 1, 1: 4}],}]
    clf_log_nosample = auto_grid_search_clf(x_train=credit_train_cut_x,
                                         y_train=credit_train_cut_y,
                                         parameter_grid=param_grid,
                                         x_test=credit_dev_x,
                                         y_test=credit_dev_y,
                                         metrics=scores)

    ## Confusion matrix
    confusion_matrix(credit_dev_y, clf_log_under.predict(credit_dev_x))

    ## Plot Performance vs. lambda
    plot_train_error(pd.Series(np.log10(param_grid[0]['C'])),
                     clf_log_nosample.cv_results_['mean_test_score'],
                     error=clf_log_nosample.cv_results_['std_test_score'],
                     xlab = 'log10(lambda)')

    ## Plot variable importance
    plot_varimp(clf_log_nosample, variable_names)

    ## Plot precision vs. recall curve
    plot_precision_recall(clf_log_nosample.decision_function(credit_dev_x), credit_dev_y)

