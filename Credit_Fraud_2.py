from Credit_Fraud_1 import *
import keras
from keras import Sequential
from keras.layers import Dense, Dropout

# Recall:

## No Sampling (Cut-only) Data:
print(credit_train_cut_x.shape)

## SMOTE SVM Data:
print(credit_train_x_smote_svm.shape)

## SMOTE borderline2 Data:
print(credit_train_x_smote.shape)


##
### Modeling SMOTE (SVM) Data
##

# Random Forest
rf_param_grid = [{'n_estimators': [500], 'criterion': ['gini'],
                 'max_features': [1, 2, 4, 8],}]
scores = ['f1_macro']

clf_rf_smote_svm = auto_grid_search_clf(x_train=credit_train_x_smote_svm,
                                       y_train=credit_train_y_smote_svm,
                                       parameter_grid=rf_param_grid,
                                       classifier=RandomForestClassifier(),
                                       x_test=credit_dev_x,
                                       y_test=credit_dev_y,
                                       metrics=scores)

## Confusion matrix
confusion_matrix(credit_dev_y, clf_rf_smote_svm.predict(credit_dev_x))

## Plot Performance vs. lambda
plot_train_error(pd.Series(rf_param_grid[0]['max_features']),
                 clf_rf_smote_svm.cv_results_['mean_test_score'],
                 error=clf_rf_smote_svm.cv_results_['std_test_score'],
                 xlab='max features')

## Plot variable importance
plot_varimp(clf_rf_smote_svm, variable_names, type ='Random Forest')

## Plot precision vs. recall curve
plot_precision_recall(clf_rf_smote_svm.predict(credit_dev_x), credit_dev_y)


# Neural Nets
model_nn_smote_svm = Sequential()
model_nn_smote_svm.add(Dense(units=64, activation='relu', input_dim=len(variable_names)))
model_nn_smote_svm.add(Dropout(0.5))
model_nn_smote_svm.add(Dense(units=16, activation='sigmoid'))
model_nn_smote_svm.add(Dropout(0.25))
model_nn_smote_svm.add(Dense(units=6, activation='sigmoid'))
model_nn_smote_svm.add(Dropout(0.125))
model_nn_smote_svm.add(Dense(units=1, activation='sigmoid'))

model_nn_smote_svm.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.rmsprop(lr=0.005),
              metrics=['accuracy', f1_score])

fit_nn_smote_svm = model_nn_smote_svm.fit(credit_train_x_smote_svm,
                   credit_train_y_smote_svm,
                   validation_split= 0.2,
                   epochs=100,
                   batch_size=512,
                   class_weight={0:1, 1:1},
                   verbose=2)

pred_nn_smote_svm = model_nn_smote_svm.predict(credit_dev_x, batch_size = 1024)

plot_nn_error(fit_nn_smote_svm)

confusion_matrix((pred_nn_smote_svm > 0.5), credit_dev_y)




# SVM
# Linear SVM
from sklearn.svm import LinearSVC, SVC
clf_svm_smote_svm = LinearSVC()
clf_svm_smote_svm.fit(credit_train_x_smote_svm, credit_train_y_smote_svm)

pred = clf_svm_smote_svm.predict(credit_dev_x)
confusion_matrix(credit_dev_y, pred)

plot_precision_recall(pred, credit_dev_y)


# Radial Basis Function kernel SVM
clf_svmRBF_smote_svm = SVC(kernel='rbf')
clf_svmRBF_smote_svm.fit(credit_train_x_smote_svm, credit_train_y_smote_svm)

pred = clf_svmRBF_smote_svm.predict(credit_dev_x)
confusion_matrix(credit_dev_y, pred)

plot_precision_recall(pred, credit_dev_y)

#####################



##
### Modeling the Cut data
##

# For the cut data: Select the optimal of:

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Implicitly weigh Class 1 4x more than class 0
rf_param_grid = [{'n_estimators': [500], 'criterion': ['gini'],
                 'max_features': [1, 2, 4, 8], 'class_weight': [{0:1, 1:4}]}]

clf_rf_nosample = auto_grid_search_clf(x_train=credit_train_cut_x,
                                       y_train=credit_train_cut_y,
                                       parameter_grid=rf_param_grid,
                                       classifier=RandomForestClassifier(),
                                       x_test=credit_dev_x,
                                       y_test=credit_dev_y,
                                       metrics=scores)

## Confusion matrix
confusion_matrix(credit_dev_y, clf_rf_nosample.predict(credit_dev_x))

## Plot Performance vs. lambda
plot_train_error(pd.Series(rf_param_grid[0]['max_features']),
                 clf_rf_nosample.cv_results_['mean_test_score'],
                 error=clf_rf_nosample.cv_results_['std_test_score'],
                 xlab='max features')

## Plot variable importance
plot_varimp(clf_rf_nosample, variable_names, type ='Random Forest')

## Plot precision vs. recall curve
plot_precision_recall(clf_rf_nosample.predict(credit_dev_x), credit_dev_y)


# Boosted Trees
from sklearn.ensemble import RandomForestClassifier

clf_rf_nosample = auto_grid_search_clf(x_train=credit_train_cut_x,
                                       y_train=credit_train_cut_y,
                                       parameter_grid=rf_param_grid,
                                       classifier=RandomForestClassifier(),
                                       x_test=credit_dev_x,
                                       y_test=credit_dev_y,
                                       metrics=scores)

## Confusion matrix
confusion_matrix(credit_dev_y, clf_rf_nosample.predict(credit_dev_x))

## Plot Performance vs. lambda
plot_train_error(pd.Series(rf_param_grid[0]['max_features']),
                 clf_rf_nosample.cv_results_['mean_test_score'],
                 error=clf_rf_nosample.cv_results_['std_test_score'],
                 xlab='max features')

## Plot variable importance
plot_varimp(clf_rf_nosample, variable_names, type ='Random Forest')

## Plot precision vs. recall curve
plot_precision_recall(clf_rf_nosample.predict(credit_dev_x), credit_dev_y)



# Neural Net
model_nn_cut = Sequential()
model_nn_cut.add(Dense(units=64, activation='relu', input_dim=len(variable_names)))
model_nn_cut.add(Dropout(0.5))
model_nn_cut.add(Dense(units=10, activation='sigmoid'))
model_nn_cut.add(Dropout(0.5))
model_nn_cut.add(Dense(units=1, activation='sigmoid'))

model_nn_cut.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.rmsprop(lr=0.001),
              metrics=['accuracy', f1_score])

fit_nn_cut = model_nn_cut.fit(credit_train_cut_x,
                       credit_train_cut_y,
                       validation_split= 0.2,
                       epochs=100,
                       batch_size=16384,
                       class_weight={0:1, 1:4})

plot_nn_error(fit_nn_cut)

pred_nn_cut = model_nn_cut.predict(credit_dev_x, batch_size = 128)
confusion_matrix((pred_nn_cut > 0.5), credit_dev_y)





######################
