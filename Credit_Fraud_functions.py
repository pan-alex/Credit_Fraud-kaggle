import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from textwrap import wrap

def hist_ecdf(var, title=''):
    import statsmodels.api as sm
    plt.subplot(121)
    plt.hist(var)
    plt.title('\n'.join(wrap(title + ' histogram', 30)))

    plt.subplot(122)
    ecdf = sm.distributions.ECDF(var)
    x = np.sort(np.linspace(min(var), max(var), num=len(var)))
    plt.step(x, ecdf(x))
    plt.title('\n'.join(wrap(title + ' eCDF', 30)))
    plt.tight_layout()
    plt.show()


def auto_grid_search_clf(x_train, y_train,
                         parameter_grid,
                         classifier=LogisticRegression(),
                         cv=5,
                         metrics=('precision', 'recall'),
                         x_test=None, y_test=None,
                         ):
    '''
    This function is a simple wrapper around sk-learn functions. It performs a
    grid search using provided parameters and prints the results.

    :param x_train: Predictors in the training data. Supplied as a np.ndarray
    :param y_train: Response variable for the training data. Supplied as a
      np.ndarray with the same length as x_train.
    :param x_test: Predictors in the dev/test data. Supplied as a np.ndarray
    :param y_test: Response variable for the dev/test data. Supplied as a
      np.ndarray with the same length as x_test.
    :param classifer: The sklearn estimator to use in classification.
    :param cv: Number, k, of cross-validation folds.
    :param scoring: Metrics to use to evaluate model performance. Supplied as a
      string or list of strings.
    :return: The set of models constructed out of the parameter grid
      Prints training metrics for each combination of parameters.
      If test data are supplied, it will apply predictions to the test data and
      print metrics of its performance.
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report

    test = False
    if y_train is not None and y_test is not None: test = True

    for score in metrics:
        clf = GridSearchCV(classifier, parameter_grid, cv=cv, scoring=score)
        clf.fit(X=x_train, y=y_train)
        print('Best Parameters')
        print(clf.best_params_)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        print(score + ': Mean, sd, Parameters')
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print('%0.4f (+/-%0.4f) for %r' % (mean, std * 2, params))

    if test:
        y_pred = clf.predict(x_test)
        print(classification_report(y_test, y_pred))

    return clf


def plot_train_error(x, y, error=None, xlab=''):
    plt.errorbar(x, y, yerr=error, fmt= '--o')
    plt.ylabel('F1 score')
    plt.xlabel(xlab)
    plt.title('Tuning Parameter vs. Performance')
    plt.show()


def get_variable_importance(model, variables, type):
    '''
    Create a df containing the coefficient for each variable
    :param model: Classifier from auto_grid_search_clf
    :param variables: pd Series of the column names from the training data
    :return: pd DataFrame containing the coefficient for each variable,
      sorted in descending order by absolute value.
    '''
    if type.lower() == 'logistic':
        coefficients = pd.Series(model.best_estimator_.coef_.reshape(-1))    # Coefficients from best (training) model
        variable_importance = pd.concat([variables, coefficients], axis=1)
    elif type.lower() == 'rf' or 'random forest':
        variable_importance = pd.Series(model.best_estimator_.feature_importances_.reshape(-1))
        variable_importance = pd.concat([variables, variable_importance], axis = 1)

    # Sort the coefficients by absolute value
    variable_importance['sort'] = variable_importance[1].abs()
    variable_importance = variable_importance.sort_values('sort', ascending=False).drop('sort', axis=1)

    return variable_importance


def plot_varimp(model, variables, type ='logistic'):
    '''
    :inputs: See get_variable_importance()
    :return: None; displays a plot of variable importance
    '''
    variable_importance = get_variable_importance(model, variables, type)

    colours = np.where(variable_importance[1] > 0, 'blue', 'red')
    plt.bar(variable_importance[0], abs(variable_importance[1]), color=colours,
            alpha = 0.5)
    plt.xticks(rotation=90)
    plt.title('Variable Importance: {}'.format(type))
    plt.show()


def plot_precision_recall(y_pred, y_test):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    average_precision = average_precision_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.step(recall, precision, alpha=0.2)
    plt.fill_between(recall, precision, step='pre', alpha=0.2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall; Average Precision:{0:0.2f}'.format(average_precision))
    plt.show()




def f1_score(y_true, y_pred):
    # from SO (https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras)
    import keras.backend as K
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def plot_nn_error(history):
    '''
    :param history: History of the trained model

    :return: Plots of F1 score and loss
    '''
    plt.subplot(211)
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.title('F1 score')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()