---
title: "2018-05-02 - Credit Card Fraud"
author: "Alex Pan"
output:
  html_document:
    keep_md: true
    theme: spacelab
    toc: yes
editor_options: 
  chunk_output_type: console
---
 
A tour of classification algorithms to detect credit card fraud, primarily using the caret package.

Data from <a href = https://www.kaggle.com/mlg-ulb/creditcardfraud/data>Kaggle</a>.




## 1. Libraries

```r
library(tidyverse)
theme_set(theme_bw())
library(caret)
library(corrplot)
library(gridExtra)


set.seed(123456)
```


## 2. Defining Functions

```r
# Chosen at random
test.class <- function(model, test=credit_dev, cl=credit_dev$Class, positive = "1") {
    # Given a model 'model', predicts the test classifier. Then returns a confusion matrix for the test data.
    # Model = A model object (ex., glm, random forest, knn)
    # test = data.frame containing all the variables used in the model
    # cl = A factor of the true classifications for the variable of interest in 'test'
    # positive = Which outcome is the outcome of interest? By default, it is 1.
   predict <- predict(model, test)
   return(confusionMatrix(predict, cl, positive = positive))
}

scaleFUN <- function(x) sprintf("%.2f", x)
```

## 3. Load in Data {.tabset}

These data are all PCA variables (i.e., these are not the actual variables)


### Data structure and Descriptive Stats:


```r
glimpse(credit)
```

```
Observations: 284,807
Variables: 31
$ Time   <int> 0, 0, 1, 1, 2, 2, 4, 7, 7, 9, 10, 10, 10, 11, 12, 12, 1...
$ V1     <dbl> -1.359807, 1.191857, -1.358354, -0.966272, -1.158233, -...
$ V2     <dbl> -0.0727812, 0.2661507, -1.3401631, -0.1852260, 0.877736...
$ V3     <dbl> 2.5363467, 0.1664801, 1.7732093, 1.7929933, 1.5487178, ...
$ V4     <dbl> 1.3781552, 0.4481541, 0.3797796, -0.8632913, 0.4030339,...
$ V5     <dbl> -0.3383208, 0.0600176, -0.5031981, -0.0103089, -0.40719...
$ V6     <dbl> 0.4623878, -0.0823608, 1.8004994, 1.2472032, 0.0959215,...
$ V7     <dbl> 0.2395986, -0.0788030, 0.7914610, 0.2376089, 0.5929407,...
$ V8     <dbl> 0.09869790, 0.08510165, 0.24767579, 0.37743587, -0.2705...
$ V9     <dbl> 0.3637870, -0.2554251, -1.5146543, -1.3870241, 0.817739...
$ V10    <dbl> 0.0907942, -0.1669744, 0.2076429, -0.0549519, 0.7530744...
$ V11    <dbl> -0.5515995, 1.6127267, 0.6245015, -0.2264873, -0.822842...
$ V12    <dbl> -0.6178009, 1.0652353, 0.0660837, 0.1782282, 0.5381956,...
$ V13    <dbl> -0.9913898, 0.4890950, 0.7172927, 0.5077569, 1.3458516,...
$ V14    <dbl> -0.3111694, -0.1437723, -0.1659459, -0.2879237, -1.1196...
$ V15    <dbl> 1.46817697, 0.63555809, 2.34586495, -0.63141812, 0.1751...
$ V16    <dbl> -0.4704005, 0.4639170, -2.8900832, -1.0596472, -0.45144...
$ V17    <dbl> 0.20797124, -0.11480466, 1.10996938, -0.68409279, -0.23...
$ V18    <dbl> 0.0257906, -0.1833613, -0.1213593, 1.9657750, -0.038194...
$ V19    <dbl> 0.4039930, -0.1457830, -2.2618571, -1.2326220, 0.803486...
$ V20    <dbl> 0.2514121, -0.0690831, 0.5249797, -0.2080378, 0.4085424...
$ V21    <dbl> -0.0183068, -0.2257752, 0.2479982, -0.1083005, -0.00943...
$ V22    <dbl> 0.2778376, -0.6386720, 0.7716794, 0.0052736, 0.7982785,...
$ V23    <dbl> -0.11047391, 0.10128802, 0.90941226, -0.19032052, -0.13...
$ V24    <dbl> 0.0669281, -0.3398465, -0.6892810, -1.1755753, 0.141267...
$ V25    <dbl> 0.1285394, 0.1671704, -0.3276418, 0.6473760, -0.2060096...
$ V26    <dbl> -0.1891148, 0.1258945, -0.1390966, -0.2219288, 0.502292...
$ V27    <dbl> 0.1335584, -0.0089831, -0.0553528, 0.0627228, 0.2194222...
$ V28    <dbl> -0.02105305, 0.01472417, -0.05975184, 0.06145763, 0.215...
$ Amount <dbl> 149.62, 2.69, 378.66, 123.50, 69.99, 3.67, 4.99, 40.80,...
$ Class  <fctr> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
```


***

### Missingness

There is one missing entry for Time. It's a non-fraud entry and it's one of 200,000--I'm just going to remove it.

```r
colSums(is.na(credit))
```

```
  Time     V1     V2     V3     V4     V5     V6     V7     V8     V9 
     1      0      0      0      0      0      0      0      0      0 
   V10    V11    V12    V13    V14    V15    V16    V17    V18    V19 
     0      0      0      0      0      0      0      0      0      0 
   V20    V21    V22    V23    V24    V25    V26    V27    V28 Amount 
     0      0      0      0      0      0      0      0      0      0 
 Class 
     0 
```


```r
credit <- na.omit(credit)
```

***

## 3. Descriptive Stats

These data are composed of anonymized variables that have been reduced by PCA. For that reason descriptive stats aren't as important to us as they would normally be, since we have absolutely no way of interpreting them.


Most transactions are very small, but some are very large. Larger transactions *may* be more associated with fraud.


```r
summary(credit$Amount)
```

```
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    0.0     5.6    22.0    88.3    77.2 25691.2 
```


```r
p1 <- ggplot(credit) +
    aes(x = Amount) +
    geom_histogram()

p2 <- ggplot(credit) +
    aes(x = Amount) +
    stat_ecdf()

grid.arrange(p1, p2, ncol = 2)
```

![](2018-05-02_Kaggle_Credit_Fraud_files/figure-html/unnamed-chunk-8-1.png)<!-- -->




Time is described as *Number of seconds elapsed between each transaction (over two days)*. I'm not 100% clear on what this means, but judging from the bimodal pattern plotted below I would suspect that data collection began around midnight and continued for 48 hours.

This is interesting because if fraud is more common at certain times of the day, we should really change this feature to 'represent time of day' rather than 'seconds since start'


```r
p1 <- ggplot(credit) +
    aes(x = Time) +
    geom_histogram() +
    scale_x_continuous(minor_breaks = seq(0, 180000, 3600))

p2 <- ggplot(credit) +
    aes(x = Time) +
    stat_ecdf()

grid.arrange(p1, p2, ncol = 2)
```

![](2018-05-02_Kaggle_Credit_Fraud_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

***

In fact, let's make a new feature now. There are a few ways to split this up--looking only at time of day, or assigning each time a score based on how many other transactions are happening around that time. E.g., breaking up transactions into 10 or 30 minute blocks and making a new variable that describes how many transactions took place in that period.

For simplicity I will just use 'time of day', which atleast roughly increases as 'time of day' increases. 

If fraud is actually associated with Time it will *probably* be non-monotonic (eg., more active at very early and very late time rather than trending up or down as 'time of day' increases). For that reason a non-linear classifier would probably do better if Time is important.



```r
credit <- credit %>%
    mutate(time_of_day = ifelse(Time < 86400, Time / 3600, (Time - 86400) / 3600))

ggplot(credit) +
    aes(x = time_of_day) +
    geom_histogram() +
    scale_x_continuous(minor_breaks = seq(0, 180000, 3600))
```

![](2018-05-02_Kaggle_Credit_Fraud_files/figure-html/unnamed-chunk-10-1.png)<!-- -->


***


## 4. Data Partitioning

1 is credit card fraud, 0 is a normal transaction.

Only 0.2% (492) of the entries are actual fraud! We will need to take special approaches to handle these unbalanced data.


```r
summary(credit$Class)    
```

```
     0      1 
284314    492 
```

***

We start by sampling data for each of the training, CV (development), and test sets. We will then undersample the non-fraud cases.

caret's `createDataPartition` function works like `sample` in base R, except it makes sure that the response variable is balanced across the different data sets. This is especially important when the number of "events" is low.


```r
mask_train <- createDataPartition(credit$Class, p = 0.7, list = F)
credit_train <- credit[mask_train, ]

# Split the remaining 30% into CV and test sets. Unfortunately this is a convoluted process
mask_cv <- createDataPartition(credit$Class[-mask_train], p = 0.5, list = F)
credit_dev <- credit[mask_cv, ]
credit_test <- credit[-mask_train, ][-mask_cv, ]
```


#### Controls for model validation

Define the controls for each model fit. Here I'm using 5x5-fold CV.

```r
fit.controls <- trainControl(method = "repeatedcv",    # Bootstrapped CV
                             number = 5,    # number of CV
                             repeats = 5)    # number of repeats
```


## 5. Dealing with Unbalanced Classes:  {.tabset}

I'm going to try and figure out which sampling methods are best for these data. For each sampling technique I will use a fairly simple random forest. If any sampling techniques seem to work better than others I will follow up on those specifically and try out other classifiers.

*(Note: For faster Rmarkdown rendering I've disabled the evaluation of the code and am just copying the output from the console)*

**Summary:** 

* Oversampling, SMOTE, and even no sampling perform much better on the dev set compared to undersampling. 

* Oversampling and No sampling produced the highest kappas in these random forest models. Unlike SMOTE and undersampling the training errors are somewhat reflective of the dev set errors, which make them more useful sampling techniques to train models on. 

* Whereas undersampling, oversampling, and SMOTE tended to overfit the training data, no sampling performed better in the dev set compared to the training set. I'm not completely sure why this is the case. (a) I could be under-trained (this will be determined later); or (b) there may be overlap between positive and negative events in the training set that precludes very high training Kappa, whereas the dev set is more sparse (?).

* Running the sampling *inside* of cross-validation might improve the agreement between the training metrics and dev metrics.

I will move forward using oversampling, SMOTE, and no sampling and try some other classification models.



### 5.1. Undersampling {.tabset}

Undersampling involves censoring a large amount of data.




```r
## Note I am excluding 'Time' and 'Amount'

credit_train_downsampled <- downSample(x = credit_train,
                               y = credit_train$Class,
                               yname = "Class")

summary(credit_train_downsampled$Class)
```

```
  0   1 
345 345 
```



```r
kappas <- data.frame(mtry = NA, set = NA, kappa = NA)

for (mtry in c(1, 2, 4, 8, 12, 16)){
    set.seed(123456)
    fit_rf <- train(Class ~ . -Time,
                    data = credit_train_downsampled,
                    method = 'rf',
                    trControl = fit.controls,
                    tuneGrid = expand.grid(mtry = mtry),
                    n.trees = 3001,
                    metric = 'Kappa')
    
    fit_metrics <- test.class(fit_rf)
    kappas <- rbind(kappas, c(mtry, 'train', fit_rf$results$Kappa))
    kappas <- rbind(kappas, c(mtry, 'dev', fit_metrics$overall['Kappa']))
    print(paste('mtry: ', mtry))
    print(fit_rf$results$Kappa)
    print(fit_metrics)
}


kappas <- kappas %>%
    na.omit() %>%
    mutate(kappa = as.numeric(kappa),
           mtry = as.numeric(mtry))
```


```r
ggplot(kappas) +
    aes(x = mtry, y = kappa, col = set) +
    geom_point() +
    geom_line() +
    scale_y_continuous(labels = scaleFUN)
```

![](undersample_rf_fit1.png)

```
1] "mtry:  1"
[1] 0.876
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42043     4
         1   584    90
                                        
               Accuracy : 0.986         
                 95% CI : (0.985, 0.987)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.231         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.95745       
            Specificity : 0.98630       
         Pos Pred Value : 0.13353       
         Neg Pred Value : 0.99990       
             Prevalence : 0.00220       
         Detection Rate : 0.00211       
   Detection Prevalence : 0.01578       
      Balanced Accuracy : 0.97187       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  2"
[1] 0.88
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 41978     3
         1   649    91
                                        
               Accuracy : 0.985         
                 95% CI : (0.984, 0.986)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.215         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.96809       
            Specificity : 0.98477       
         Pos Pred Value : 0.12297       
         Neg Pred Value : 0.99993       
             Prevalence : 0.00220       
         Detection Rate : 0.00213       
   Detection Prevalence : 0.01732       
      Balanced Accuracy : 0.97643       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  4"
[1] 0.891
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 41838     2
         1   789    92
                                       
               Accuracy : 0.981        
                 95% CI : (0.98, 0.983)
    No Information Rate : 0.998        
    P-Value [Acc > NIR] : 1            
                                       
                  Kappa : 0.185        
 Mcnemar's Test P-Value : <2e-16       
                                       
            Sensitivity : 0.97872      
            Specificity : 0.98149      
         Pos Pred Value : 0.10443      
         Neg Pred Value : 0.99995      
             Prevalence : 0.00220      
         Detection Rate : 0.00215      
   Detection Prevalence : 0.02062      
      Balanced Accuracy : 0.98011      
                                       
       'Positive' Class : 1            
                                       
[1] "mtry:  8"
[1] 0.89
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 41674     2
         1   953    92
                                        
               Accuracy : 0.978         
                 95% CI : (0.976, 0.979)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.158         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.97872       
            Specificity : 0.97764       
         Pos Pred Value : 0.08804       
         Neg Pred Value : 0.99995       
             Prevalence : 0.00220       
         Detection Rate : 0.00215       
   Detection Prevalence : 0.02446       
      Balanced Accuracy : 0.97818       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  12"
[1] 0.888
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 41554     2
         1  1073    92
                                        
               Accuracy : 0.975         
                 95% CI : (0.973, 0.976)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.143         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.97872       
            Specificity : 0.97483       
         Pos Pred Value : 0.07897       
         Neg Pred Value : 0.99995       
             Prevalence : 0.00220       
         Detection Rate : 0.00215       
   Detection Prevalence : 0.02727       
      Balanced Accuracy : 0.97678       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  16"
[1] 0.885
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 41470     2
         1  1157    92
                                        
               Accuracy : 0.973         
                 95% CI : (0.971, 0.974)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.133         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.97872       
            Specificity : 0.97286       
         Pos Pred Value : 0.07366       
         Neg Pred Value : 0.99995       
             Prevalence : 0.00220       
         Detection Rate : 0.00215       
   Detection Prevalence : 0.02924       
      Balanced Accuracy : 0.97579       
                                        
       'Positive' Class : 1        
```


The accuracy is very high, but that is to be expected when the classes are so imbalanced. Since only 0.2% of the observations are positive events, an accuracy of 99.8% is the No Information Rate.

A useful statistic is Kappa, since it takes into account the marginal distribution of the response variable, or the positive predictive value (precision) vs. recall (sensitivity).

Using these metrics, undersampling produces *reasonable* (but not amazing) results.

***

### 5.2. Oversampling

Undersampling may not work so well when we don't have a lot of positive events, since it throws away much data. Still, I don't want to have a huge amount of duplicated data, so I'm going to partially down-sample first.



```r
temp <- credit_train %>%
    filter(Class == 0) %>%
    sample_n(10000)

temp2 <- credit_train %>%
    filter(Class == 1)

credit_10000 <- rbind(temp, temp2)

credit_train_oversampled <- upSample(x = credit_10000,
                                     y = credit_10000$Class,
                                     yname = 'Class')

summary(credit_train_oversampled$Class)
```

```
    0     1 
10000 10000 
```


```r
kappas <- data.frame(mtry = NA, set = NA, kappa = NA)

for (mtry in c(1, 2, 4, 8, 12, 16)){
    set.seed(123456)
    fit_rf <- train(Class ~ . -Time,
                    data = credit_train_oversampled,
                    method = 'rf',
                    trControl = fit.controls,
                    tuneGrid = expand.grid(mtry = mtry),
                    n.trees = 3001,
                    metric = 'Kappa')
    
    fit_metrics <- test.class(fit_rf)
    kappas <- rbind(kappas, c(mtry, 'train', fit_rf$results$Kappa))
    kappas <- rbind(kappas, c(mtry, 'dev', fit_metrics$overall['Kappa']))
    print(paste('mtry: ', mtry))
    print(fit_rf$results$Kappa)
    print(fit_metrics)
}


kappas <- kappas %>%
    na.omit() %>%
    mutate(kappa = as.numeric(kappa),
           mtry = as.numeric(mtry))
```


```r
ggplot(kappas) +
    aes(x = mtry, y = kappa, col = set) +
    geom_point() +
    geom_line()
    scale_y_continuous(labels = scaleFUN)
```

![](oversample_rf_fit1.png)

```
[1] "mtry:  1"
[1] 1
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42605     6
         1    22    88
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : 1.17e-15  
                                    
                  Kappa : 0.862     
 Mcnemar's Test P-Value : 0.00459   
                                    
            Sensitivity : 0.93617   
            Specificity : 0.99948   
         Pos Pred Value : 0.80000   
         Neg Pred Value : 0.99986   
             Prevalence : 0.00220   
         Detection Rate : 0.00206   
   Detection Prevalence : 0.00257   
      Balanced Accuracy : 0.96783   
                                    
       'Positive' Class : 1         
                                    
[1] "mtry:  2"
[1] 0.999
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42602     5
         1    25    89
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : 1.23e-14  
                                    
                  Kappa : 0.855     
 Mcnemar's Test P-Value : 0.000523  
                                    
            Sensitivity : 0.94681   
            Specificity : 0.99941   
         Pos Pred Value : 0.78070   
         Neg Pred Value : 0.99988   
             Prevalence : 0.00220   
         Detection Rate : 0.00208   
   Detection Prevalence : 0.00267   
      Balanced Accuracy : 0.97311   
                                    
       'Positive' Class : 1         
                                    
[1] "mtry:  4"
[1] 0.999
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42601     5
         1    26    89
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : 3.79e-14  
                                    
                  Kappa : 0.851     
 Mcnemar's Test P-Value : 0.000328  
                                    
            Sensitivity : 0.94681   
            Specificity : 0.99939   
         Pos Pred Value : 0.77391   
         Neg Pred Value : 0.99988   
             Prevalence : 0.00220   
         Detection Rate : 0.00208   
   Detection Prevalence : 0.00269   
      Balanced Accuracy : 0.97310   
                                    
       'Positive' Class : 1         
                                    
[1] "mtry:  8"
[1] 0.999
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42602     5
         1    25    89
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : 1.23e-14  
                                    
                  Kappa : 0.855     
 Mcnemar's Test P-Value : 0.000523  
                                    
            Sensitivity : 0.94681   
            Specificity : 0.99941   
         Pos Pred Value : 0.78070   
         Neg Pred Value : 0.99988   
             Prevalence : 0.00220   
         Detection Rate : 0.00208   
   Detection Prevalence : 0.00267   
      Balanced Accuracy : 0.97311   
                                    
       'Positive' Class : 1         
                                    
[1] "mtry:  12"
[1] 0.999
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42602     5
         1    25    89
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : 1.23e-14  
                                    
                  Kappa : 0.855     
 Mcnemar's Test P-Value : 0.000523  
                                    
            Sensitivity : 0.94681   
            Specificity : 0.99941   
         Pos Pred Value : 0.78070   
         Neg Pred Value : 0.99988   
             Prevalence : 0.00220   
         Detection Rate : 0.00208   
   Detection Prevalence : 0.00267   
      Balanced Accuracy : 0.97311   
                                    
       'Positive' Class : 1         
                                    
[1] "mtry:  16"
[1] 0.999
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42601     5
         1    26    89
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : 3.79e-14  
                                    
                  Kappa : 0.851     
 Mcnemar's Test P-Value : 0.000328  
                                    
            Sensitivity : 0.94681   
            Specificity : 0.99939   
         Pos Pred Value : 0.77391   
         Neg Pred Value : 0.99988   
             Prevalence : 0.00220   
         Detection Rate : 0.00208   
   Detection Prevalence : 0.00269   
      Balanced Accuracy : 0.97310   
                                    
       'Positive' Class : 1         
```

***

### 5.3 SMOTE

Synthetic minority over-sampling technique is a hybrid method that undersamples the majority class and oversamples the minority class. Rather than being exact replicates, the minority observations are imputed by interpolating values from other minority class observations (using k nearest neighbours).



```r
# Note: SMOTE only takes Dataframes (Does not accept tbls)
# The default sampling is 200% over and 200% under, and k = 5
credit_train_smote <- DMwR::SMOTE(Class ~ . - Time, 
                                  data = as.data.frame(credit_train))
summary(credit_train_smote$Class)
```

```
   0    1 
1380 1035 
```


```r
kappas <- data.frame(mtry = NA, set = NA, kappa = NA)

for (mtry in c(1, 2, 4, 8, 12, 16)){
    set.seed(123456)
    fit_rf <- train(Class ~ . -Time,
                    data = credit_train_smote,
                    method = 'rf',
                    trControl = fit.controls,
                    tuneGrid = expand.grid(mtry = mtry),
                    n.trees = 3001,
                    metric = 'Kappa')
    
    fit_metrics <- test.class(fit_rf)
    kappas <- rbind(kappas, c(mtry, 'train', fit_rf$results$Kappa))
    kappas <- rbind(kappas, c(mtry, 'dev', fit_metrics$overall['Kappa']))
    print(paste('mtry: ', mtry))
    print(fit_rf$results$Kappa)
    print(fit_metrics)
}


kappas <- kappas %>%
    na.omit() %>%
    mutate(kappa = as.numeric(kappa),
           mtry = as.numeric(mtry))
```


```r
ggplot(kappas) +
    aes(x = mtry, y = kappa, col = set) +
    geom_point() +
    geom_line()
    scale_y_continuous(labels = scaleFUN)
```

***

SMOTE appears to have some potential. The performance when `mtry = 1` is decent. It is definitely overfiting the training data.


![](smote_fit_rf1.png)



```
[1] "mtry:  1"
[1] 0.937
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42460     5
         1   167    89
                                        
               Accuracy : 0.996         
                 95% CI : (0.995, 0.997)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.507         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99608       
         Pos Pred Value : 0.34766       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.00599       
      Balanced Accuracy : 0.97145       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  2"
[1] 0.94
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42396     5
         1   231    89
                                        
               Accuracy : 0.994         
                 95% CI : (0.994, 0.995)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.428         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99458       
         Pos Pred Value : 0.27812       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.00749       
      Balanced Accuracy : 0.97069       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  4"
[1] 0.944
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42271     5
         1   356    89
                                        
               Accuracy : 0.992         
                 95% CI : (0.991, 0.992)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.328         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99165       
         Pos Pred Value : 0.20000       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.01042       
      Balanced Accuracy : 0.96923       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  8"
[1] 0.945
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42170     5
         1   457    89
                                       
               Accuracy : 0.989        
                 95% CI : (0.988, 0.99)
    No Information Rate : 0.998        
    P-Value [Acc > NIR] : 1            
                                       
                  Kappa : 0.275        
 Mcnemar's Test P-Value : <2e-16       
                                       
            Sensitivity : 0.94681      
            Specificity : 0.98928      
         Pos Pred Value : 0.16300      
         Neg Pred Value : 0.99988      
             Prevalence : 0.00220      
         Detection Rate : 0.00208      
   Detection Prevalence : 0.01278      
      Balanced Accuracy : 0.96804      
                                       
       'Positive' Class : 1            
                                       
[1] "mtry:  12"
[1] 0.947
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42085     4
         1   542    90
                                        
               Accuracy : 0.987         
                 95% CI : (0.986, 0.988)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.245         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.95745       
            Specificity : 0.98729       
         Pos Pred Value : 0.14241       
         Neg Pred Value : 0.99990       
             Prevalence : 0.00220       
         Detection Rate : 0.00211       
   Detection Prevalence : 0.01479       
      Balanced Accuracy : 0.97237       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  16"
[1] 0.946
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42047     2
         1   580    92
                                        
               Accuracy : 0.986         
                 95% CI : (0.985, 0.987)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1             
                                        
                  Kappa : 0.237         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.97872       
            Specificity : 0.98639       
         Pos Pred Value : 0.13690       
         Neg Pred Value : 0.99995       
             Prevalence : 0.00220       
         Detection Rate : 0.00215       
   Detection Prevalence : 0.01573       
      Balanced Accuracy : 0.98256       
                                        
       'Positive' Class : 1     
```


***

### 5.4 SMOTE v 2

I think the main reason why SMOTE is performing poorly compared to straight-up oversampling is the number of training observations. I'm going to try anoher random forest with SMOTE, but create a lot more minority cases.



```r
# Choose parameters that give roughly the same number and ~ 20,000 observations
credit_train_smote2 <- DMwR::SMOTE(Class ~ . - Time, 
                                  data = as.data.frame(credit_train),
                                  perc.over = 2700,
                                  perc.under = 110) # Does not accept tbls
summary(credit_train_smote2$Class)
```

```
    0     1 
10246  9660 
```


```r
kappas <- data.frame(mtry = NA, set = NA, kappa = NA)

for (mtry in c(1, 2, 4, 8, 12, 16)){
    set.seed(123456)
    fit_rf <- train(Class ~ . -Time,
                    data = credit_train_smote2,
                    method = 'rf',
                    trControl = fit.controls,
                    tuneGrid = expand.grid(mtry = mtry),
                    n.trees = 3001,
                    metric = 'Kappa')
    
    fit_metrics <- test.class(fit_rf)
    kappas <- rbind(kappas, c(mtry, 'train', fit_rf$results$Kappa))
    kappas <- rbind(kappas, c(mtry, 'dev', fit_metrics$overall['Kappa']))
    print(paste('mtry: ', mtry))
    print(fit_rf$results$Kappa)
    print(fit_metrics)
}


kappas <- kappas %>%
    na.omit() %>%
    mutate(kappa = as.numeric(kappa),
           mtry = as.numeric(mtry))
```


```r
ggplot(kappas) +
    aes(x = mtry, y = kappa, col = set) +
    geom_point() +
    geom_line() + 
    scale_y_continuous(labels = scaleFUN)
```


![](smote_fit_rf2.png)

```
[1] "mtry:  1"
[1] 0.989
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42589     5
         1    38    89
                                        
               Accuracy : 0.999         
                 95% CI : (0.999, 0.999)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 3.06e-09      
                                        
                  Kappa : 0.805         
 Mcnemar's Test P-Value : 1.06e-06      
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99911       
         Pos Pred Value : 0.70079       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.00297       
      Balanced Accuracy : 0.97296       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  2"
[1] 0.991
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42587     5
         1    40    89
                                        
               Accuracy : 0.999         
                 95% CI : (0.999, 0.999)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1.42e-08      
                                        
                  Kappa : 0.798         
 Mcnemar's Test P-Value : 4.01e-07      
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99906       
         Pos Pred Value : 0.68992       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.00302       
      Balanced Accuracy : 0.97294       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  4"
[1] 0.991
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42579     5
         1    48    89
                                        
               Accuracy : 0.999         
                 95% CI : (0.998, 0.999)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 2.89e-06      
                                        
                  Kappa : 0.77          
 Mcnemar's Test P-Value : 7.97e-09      
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99887       
         Pos Pred Value : 0.64964       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.00321       
      Balanced Accuracy : 0.97284       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  8"
[1] 0.99
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42565     5
         1    62    89
                                        
               Accuracy : 0.998         
                 95% CI : (0.998, 0.999)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 0.00209       
                                        
                  Kappa : 0.726         
 Mcnemar's Test P-Value : 7.84e-12      
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99855       
         Pos Pred Value : 0.58940       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.00353       
      Balanced Accuracy : 0.97268       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  12"
[1] 0.99
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42546     5
         1    81    89
                                        
               Accuracy : 0.998         
                 95% CI : (0.998, 0.998)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 0.221         
                                        
                  Kappa : 0.673         
 Mcnemar's Test P-Value : 6.09e-16      
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99810       
         Pos Pred Value : 0.52353       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.00398       
      Balanced Accuracy : 0.97245       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  16"
[1] 0.989
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42535     5
         1    92    89
                                        
               Accuracy : 0.998         
                 95% CI : (0.997, 0.998)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 0.647         
                                        
                  Kappa : 0.646         
 Mcnemar's Test P-Value : <2e-16        
                                        
            Sensitivity : 0.94681       
            Specificity : 0.99784       
         Pos Pred Value : 0.49171       
         Neg Pred Value : 0.99988       
             Prevalence : 0.00220       
         Detection Rate : 0.00208       
   Detection Prevalence : 0.00424       
      Balanced Accuracy : 0.97233       
                                        
       'Positive' Class : 1             
                                        
```


### 5.5 No Sampling

I wasn't going to run this initially because sampling methods are supposed to help improve training performance. However, I have noticed that models performing better on the training data tend to perform worse on the dev set. 

Perhaps they have been overfitted, but it is also possible that the large disparity in prevalence of fraud between the training set and dev set make it hard to pick the best training model in advance. 


```r
kappas <- data.frame(mtry = NA, set = NA, kappa = NA)

for (mtry in c(1, 2, 4, 8, 12, 16)){
    set.seed(123456)
    fit_rf <- train(Class ~ . -Time,
                    data = sample_n(credit_train, 20000),
                    method = 'rf',
                    trControl = fit.controls,
                    tuneGrid = expand.grid(mtry = mtry),
                    n.trees = 3001,
                    metric = 'Kappa')
    
    fit_metrics <- test.class(fit_rf)
    kappas <- rbind(kappas, c(mtry, 'train', fit_rf$results$Kappa))
    kappas <- rbind(kappas, c(mtry, 'dev', fit_metrics$overall['Kappa']))
    print(paste('mtry: ', mtry))
    print(fit_metrics)
}


kappas <- kappas %>%
    na.omit() %>%
    mutate(kappa = as.numeric(kappa),
           mtry = as.numeric(mtry))
```



```r
ggplot(kappas) +
    aes(x = mtry, y = kappa, col = set) +
    geom_point() +
    geom_line()
    scale_y_continuous(labels = scaleFUN)
```

![](no_sample_fit_rf1.png)

```
[1] "mtry:  1"
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42626    44
         1     1    50
                                        
               Accuracy : 0.999         
                 95% CI : (0.999, 0.999)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 1.42e-08      
                                        
                  Kappa : 0.689         
 Mcnemar's Test P-Value : 3.83e-10      
                                        
            Sensitivity : 0.53191       
            Specificity : 0.99998       
         Pos Pred Value : 0.98039       
         Neg Pred Value : 0.99897       
             Prevalence : 0.00220       
         Detection Rate : 0.00117       
   Detection Prevalence : 0.00119       
      Balanced Accuracy : 0.76595       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  2"
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42625    32
         1     2    62
                                        
               Accuracy : 0.999         
                 95% CI : (0.999, 0.999)
    No Information Rate : 0.998         
    P-Value [Acc > NIR] : 9.23e-13      
                                        
                  Kappa : 0.784         
 Mcnemar's Test P-Value : 6.58e-07      
                                        
            Sensitivity : 0.65957       
            Specificity : 0.99995       
         Pos Pred Value : 0.96875       
         Neg Pred Value : 0.99925       
             Prevalence : 0.00220       
         Detection Rate : 0.00145       
   Detection Prevalence : 0.00150       
      Balanced Accuracy : 0.82976       
                                        
       'Positive' Class : 1             
                                        
[1] "mtry:  4"
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42625    26
         1     2    68
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : 1.17e-15  
                                    
                  Kappa : 0.829     
 Mcnemar's Test P-Value : 1.38e-05  
                                    
            Sensitivity : 0.72340   
            Specificity : 0.99995   
         Pos Pred Value : 0.97143   
         Neg Pred Value : 0.99939   
             Prevalence : 0.00220   
         Detection Rate : 0.00159   
   Detection Prevalence : 0.00164   
      Balanced Accuracy : 0.86168   
                                    
       'Positive' Class : 1         
                                    
[1] "mtry:  8"
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42625    22
         1     2    72
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : < 2e-16   
                                    
                  Kappa : 0.857     
 Mcnemar's Test P-Value : 0.000105  
                                    
            Sensitivity : 0.76596   
            Specificity : 0.99995   
         Pos Pred Value : 0.97297   
         Neg Pred Value : 0.99948   
             Prevalence : 0.00220   
         Detection Rate : 0.00169   
   Detection Prevalence : 0.00173   
      Balanced Accuracy : 0.88296   
                                    
       'Positive' Class : 1         
                                    
[1] "mtry:  12"
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42626    23
         1     1    71
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : < 2e-16   
                                    
                  Kappa : 0.855     
 Mcnemar's Test P-Value : 1.81e-05  
                                    
            Sensitivity : 0.75532   
            Specificity : 0.99998   
         Pos Pred Value : 0.98611   
         Neg Pred Value : 0.99946   
             Prevalence : 0.00220   
         Detection Rate : 0.00166   
   Detection Prevalence : 0.00169   
      Balanced Accuracy : 0.87765   
                                    
       'Positive' Class : 1         
                                    
[1] "mtry:  16"
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 42623    23
         1     4    71
                                    
               Accuracy : 0.999     
                 95% CI : (0.999, 1)
    No Information Rate : 0.998     
    P-Value [Acc > NIR] : 3.43e-16  
                                    
                  Kappa : 0.84      
 Mcnemar's Test P-Value : 0.000532  
                                    
            Sensitivity : 0.75532   
            Specificity : 0.99991   
         Pos Pred Value : 0.94667   
         Neg Pred Value : 0.99946   
             Prevalence : 0.00220   
         Detection Rate : 0.00166   
   Detection Prevalence : 0.00176   
      Balanced Accuracy : 0.87761   
                                    
       'Positive' Class : 1    
```



***

## 6. Classification {.tabset}

In this section I will do some parameter searches across a number of classification models using either Oversampled or SMOTE-sampled data. I'll check the performance against the dev set and select a few candidate models.

In a later section I'll manually play with each of the most promising models and try to eek out additional performance before finally testing 1-3 models on the test set to get an unbiased estimate of error.

<!-- ### Notes -->

<!-- I found that the training error tends to be lower for a flexible model whereas the training error is higher for a rigid model (eg. using a small lambda in penalized logistic regression vs a large one).  -->

<!-- However, it seems that (relatively) rigid models have better performance in the dev set. This probably stems from the difference in prevalence between the training set (artificially balanced) and the dev set (reflects nature)---both accuracy and Kappa are not good metrics if the training data are balanced but the CV data are not. -->

<!-- At one point I also tried using the F1 score, but I think the underlying problem is that the training data are balanced (hence F1, Kappa, and Accuracy will be similar for a given model) whereas the CV data are not (hence F1 and Kappa differ from Accuracy). -->


<!-- ### 6.1 Oversampling {.tabset} -->




<!-- ### 6.2 SMOTE {.tabset} -->


<!-- #### 6.2.1 Logistic Regression (Stepwise) -->

<!-- <!-- Logistic regression using all of the features has very poor PPV / Kappa. --> -->
<!-- <!-- ```{r, results = 'hide'} --> -->
<!-- <!-- set.seed(123456) --> -->
<!-- <!-- fit_logistic <- train(Class ~ . - Time, --> -->
<!-- <!--                       data = credit_train_smote, --> -->
<!-- <!--                       method = 'glmStepAIC', --> -->
<!-- <!--                       direction = 'both', --> -->
<!-- <!--                       trControl = fit.controls) --> -->
<!-- <!-- ``` --> -->

<!-- <!-- ```{r} --> -->
<!-- <!-- test.class(fit_logistic) --> -->
<!-- <!-- ``` --> -->

<!-- <!-- ```{r} --> -->
<!-- <!-- plot(varImp(fit_logistic)) --> -->
<!-- <!-- ``` --> -->


<!-- #### 6.2.2 Penalized Logistic Regression -->

<!-- Very high penalties give better PPV, but sensitivity suffers. Of the parameters tested here, a lambda of 1000 and 10,000 provide a good balance of PPV and sensitivity. A lambda of 100,000 results in a model that is undertrained and misses a lot of positive examples. -->

<!-- ```{r, eval = FALSE} -->
<!-- # Tested 0.001 to 10^5 -->
<!-- # Wrapped in a for loop instead of built-in train because I want to see the CV errors for each model instead of jus the most performant on the training set. -->

<!-- for (lambda in c(1, 100, 1000, 10^4, 10^5)){ -->
<!--     grid_plr <- expand.grid(lambda = lambda, cp = 'bic') -->
<!--     set.seed(123456) -->
<!--     fit_logistic <- train(Class ~ . - Time, -->
<!--                           data = credit_train_smote, -->
<!--                           method = 'plr', -->
<!--                           trControl = F1_metric, -->
<!--                           tuneGrid = grid_plr, -->
<!--                           metric = 'F1') -->

<!--     print(lambda) -->
<!--     print(test.class(fit_logistic)) -->
<!-- } -->
<!-- ``` -->


<!-- Lambda = 1 -->

<!-- ``` -->
<!-- [1] 1 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 41901    10 -->
<!--          1   719    91 -->

<!--                Accuracy : 0.983          -->
<!--                  95% CI : (0.982, 0.984) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 1              -->

<!--                   Kappa : 0.196          -->
<!--  Mcnemar's Test P-Value : <2e-16         -->

<!--             Sensitivity : 0.90099        -->
<!--             Specificity : 0.98313        -->
<!--          Pos Pred Value : 0.11235        -->
<!--          Neg Pred Value : 0.99976        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00213        -->
<!--    Detection Prevalence : 0.01896        -->
<!--       Balanced Accuracy : 0.94206        -->

<!--        'Positive' Class : 1          -->
<!-- ``` -->

<!-- Lambda = 100 -->

<!-- ``` -->
<!-- [1] 100 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 42168    12 -->
<!--          1   452    89 -->

<!--                Accuracy : 0.989         -->
<!--                  95% CI : (0.988, 0.99) -->
<!--     No Information Rate : 0.998         -->
<!--     P-Value [Acc > NIR] : 1             -->

<!--                   Kappa : 0.274         -->
<!--  Mcnemar's Test P-Value : <2e-16        -->

<!--             Sensitivity : 0.88119       -->
<!--             Specificity : 0.98939       -->
<!--          Pos Pred Value : 0.16451       -->
<!--          Neg Pred Value : 0.99972       -->
<!--              Prevalence : 0.00236       -->
<!--          Detection Rate : 0.00208       -->
<!--    Detection Prevalence : 0.01266       -->
<!--       Balanced Accuracy : 0.93529       -->

<!--        'Positive' Class : 1             -->
<!-- ``` -->

<!-- Lambda = 1000 -->

<!-- ``` -->
<!-- [1] 1000 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 42598    16 -->
<!--          1    22    85 -->

<!--                Accuracy : 0.999          -->
<!--                  95% CI : (0.999, 0.999) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 5.79e-13       -->

<!--                   Kappa : 0.817          -->
<!--  Mcnemar's Test P-Value : 0.417          -->

<!--             Sensitivity : 0.84158        -->
<!--             Specificity : 0.99948        -->
<!--          Pos Pred Value : 0.79439        -->
<!--          Neg Pred Value : 0.99962        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00199        -->
<!--    Detection Prevalence : 0.00250        -->
<!--       Balanced Accuracy : 0.92053        -->

<!--        'Positive' Class : 1     -->
<!-- ``` -->

<!-- Lambda = 10,000 -->

<!-- ``` -->
<!-- [1] 10000 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 42605    19 -->
<!--          1    15    82 -->

<!--                Accuracy : 0.999          -->
<!--                  95% CI : (0.999, 0.999) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 9.23e-15       -->

<!--                   Kappa : 0.828          -->
<!--  Mcnemar's Test P-Value : 0.607          -->

<!--             Sensitivity : 0.81188        -->
<!--             Specificity : 0.99965        -->
<!--          Pos Pred Value : 0.84536        -->
<!--          Neg Pred Value : 0.99955        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00192        -->
<!--    Detection Prevalence : 0.00227        -->
<!--       Balanced Accuracy : 0.90576        -->

<!--        'Positive' Class : 1              -->
<!-- ``` -->

<!-- Lambda = 100,000 -->

<!-- ``` -->
<!-- [1] 1e+05 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 42606    47 -->
<!--          1    14    54 -->

<!--                Accuracy : 0.999          -->
<!--                  95% CI : (0.998, 0.999) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 1.18e-05       -->

<!--                   Kappa : 0.638          -->
<!--  Mcnemar's Test P-Value : 4.18e-05       -->

<!--             Sensitivity : 0.53465        -->
<!--             Specificity : 0.99967        -->
<!--          Pos Pred Value : 0.79412        -->
<!--          Neg Pred Value : 0.99890        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00126        -->
<!--    Detection Prevalence : 0.00159        -->
<!--       Balanced Accuracy : 0.76716        -->

<!--        'Positive' Class : 1              -->

<!-- ``` -->

<!-- #### 6.2.3 KNN -->

<!-- The performance of KNN is poor. This doesn't come as a huge surprise because positive events are so sparse in the dev set. Just by the prevalence, the *k nearest neighbours* to a given positive observation will overwhelmingly be negative observations. But once again, the models are trained on data with equal prevalence. -->

<!-- The best performance came with k = 401, but this has poor sensitivity, which is likewise unsurprising. The jump from k = 351 to k = 401 is somewhat surprising. But since this isn't likely to be a worthwhile model I'm not going to pursue that. -->

<!-- ```{r} -->

<!-- # Transform these variables into Z scores for KNN -->
<!-- credit_train_smote_normalized <- credit_train_smote %>% -->
<!--     mutate(time_of_day = (time_of_day - mean(time_of_day) / sd(time_of_day)), -->
<!--            Amount = (Amount - mean(Amount) / sd(Amount))) -->


<!-- for (k in seq(1, 501, by = 50)){ -->
<!--     grid_knn <- expand.grid(k = k) -->
<!--     fit_knn <- train(Class ~ . - Time - time_of_day - Amount, -->
<!--                      data = credit_train_smote_normalized, -->
<!--                      method = "knn", -->
<!--                      tuneGrid = grid_knn, -->
<!--                      trControl = fit.controls, -->
<!--                      metric = "Kappa") -->

<!--     print(k) -->
<!--     print(test.class(fit_knn)) -->
<!-- } -->
<!-- ``` -->

<!-- ``` -->
<!-- [1] 1 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 40264     3 -->
<!--          1  2356    98 -->

<!--                Accuracy : 0.945          -->
<!--                  95% CI : (0.943, 0.947) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 1              -->

<!--                   Kappa : 0.073          -->
<!--  Mcnemar's Test P-Value : <2e-16         -->

<!--             Sensitivity : 0.97030        -->
<!--             Specificity : 0.94472        -->
<!--          Pos Pred Value : 0.03993        -->
<!--          Neg Pred Value : 0.99993        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00229        -->
<!--    Detection Prevalence : 0.05744        -->
<!--       Balanced Accuracy : 0.95751        -->

<!--        'Positive' Class : 1              -->

<!-- [1] 51 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 40122    21 -->
<!--          1  2498    80 -->

<!--                Accuracy : 0.941          -->
<!--                  95% CI : (0.939, 0.943) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 1              -->

<!--                   Kappa : 0.055          -->
<!--  Mcnemar's Test P-Value : <2e-16         -->

<!--             Sensitivity : 0.79208        -->
<!--             Specificity : 0.94139        -->
<!--          Pos Pred Value : 0.03103        -->
<!--          Neg Pred Value : 0.99948        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00187        -->
<!--    Detection Prevalence : 0.06035        -->
<!--       Balanced Accuracy : 0.86673        -->

<!--        'Positive' Class : 1              -->

<!-- [1] 101 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 39461    26 -->
<!--          1  3159    75 -->

<!--                Accuracy : 0.925          -->
<!--                  95% CI : (0.923, 0.928) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 1              -->

<!--                   Kappa : 0.041          -->
<!--  Mcnemar's Test P-Value : <2e-16         -->

<!--             Sensitivity : 0.74257        -->
<!--             Specificity : 0.92588        -->
<!--          Pos Pred Value : 0.02319        -->
<!--          Neg Pred Value : 0.99934        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00176        -->
<!--    Detection Prevalence : 0.07570        -->
<!--       Balanced Accuracy : 0.83423        -->

<!--        'Positive' Class : 1              -->

<!-- [1] 151 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 39355    28 -->
<!--          1  3265    73 -->

<!--                Accuracy : 0.923         -->
<!--                  95% CI : (0.92, 0.925) -->
<!--     No Information Rate : 0.998         -->
<!--     P-Value [Acc > NIR] : 1             -->

<!--                   Kappa : 0.038         -->
<!--  Mcnemar's Test P-Value : <2e-16        -->

<!--             Sensitivity : 0.72277       -->
<!--             Specificity : 0.92339       -->
<!--          Pos Pred Value : 0.02187       -->
<!--          Neg Pred Value : 0.99929       -->
<!--              Prevalence : 0.00236       -->
<!--          Detection Rate : 0.00171       -->
<!--    Detection Prevalence : 0.07813       -->
<!--       Balanced Accuracy : 0.82308       -->

<!--        'Positive' Class : 1             -->

<!-- [1] 201 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 40361    32 -->
<!--          1  2259    69 -->

<!--                Accuracy : 0.946          -->
<!--                  95% CI : (0.944, 0.948) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 1              -->

<!--                   Kappa : 0.053          -->
<!--  Mcnemar's Test P-Value : <2e-16         -->

<!--             Sensitivity : 0.68317        -->
<!--             Specificity : 0.94700        -->
<!--          Pos Pred Value : 0.02964        -->
<!--          Neg Pred Value : 0.99921        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00162        -->
<!--    Detection Prevalence : 0.05449        -->
<!--       Balanced Accuracy : 0.81508        -->

<!--        'Positive' Class : 1              -->

<!-- [1] 251 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 40816    34 -->
<!--          1  1804    67 -->

<!--                Accuracy : 0.957          -->
<!--                  95% CI : (0.955, 0.959) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 1              -->

<!--                   Kappa : 0.064          -->
<!--  Mcnemar's Test P-Value : <2e-16         -->

<!--             Sensitivity : 0.66337        -->
<!--             Specificity : 0.95767        -->
<!--          Pos Pred Value : 0.03581        -->
<!--          Neg Pred Value : 0.99917        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00157        -->
<!--    Detection Prevalence : 0.04380        -->
<!--       Balanced Accuracy : 0.81052        -->

<!--        'Positive' Class : 1              -->

<!-- [1] 301 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 40864    36 -->
<!--          1  1756    65 -->

<!--                Accuracy : 0.958         -->
<!--                  95% CI : (0.956, 0.96) -->
<!--     No Information Rate : 0.998         -->
<!--     P-Value [Acc > NIR] : 1             -->

<!--                   Kappa : 0.063         -->
<!--  Mcnemar's Test P-Value : <2e-16        -->

<!--             Sensitivity : 0.64356       -->
<!--             Specificity : 0.95880       -->
<!--          Pos Pred Value : 0.03569       -->
<!--          Neg Pred Value : 0.99912       -->
<!--              Prevalence : 0.00236       -->
<!--          Detection Rate : 0.00152       -->
<!--    Detection Prevalence : 0.04263       -->
<!--       Balanced Accuracy : 0.80118       -->

<!--        'Positive' Class : 1             -->

<!-- [1] 351 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 41434    39 -->
<!--          1  1186    62 -->

<!--                Accuracy : 0.971         -->
<!--                  95% CI : (0.97, 0.973) -->
<!--     No Information Rate : 0.998         -->
<!--     P-Value [Acc > NIR] : 1             -->

<!--                   Kappa : 0.088         -->
<!--  Mcnemar's Test P-Value : <2e-16        -->

<!--             Sensitivity : 0.61386       -->
<!--             Specificity : 0.97217       -->
<!--          Pos Pred Value : 0.04968       -->
<!--          Neg Pred Value : 0.99906       -->
<!--              Prevalence : 0.00236       -->
<!--          Detection Rate : 0.00145       -->
<!--    Detection Prevalence : 0.02921       -->
<!--       Balanced Accuracy : 0.79302       -->

<!--        'Positive' Class : 1             -->

<!-- [1] 401 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 42608    48 -->
<!--          1    12    53 -->

<!--                Accuracy : 0.999          -->
<!--                  95% CI : (0.998, 0.999) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 6.99e-06       -->

<!--                   Kappa : 0.638          -->
<!--  Mcnemar's Test P-Value : 6.23e-06       -->

<!--             Sensitivity : 0.52475        -->
<!--             Specificity : 0.99972        -->
<!--          Pos Pred Value : 0.81538        -->
<!--          Neg Pred Value : 0.99887        -->
<!--              Prevalence : 0.00236        -->
<!--          Detection Rate : 0.00124        -->
<!--    Detection Prevalence : 0.00152        -->
<!--       Balanced Accuracy : 0.76224        -->

<!--        'Positive' Class : 1 -->

<!-- [1] 451 -->
<!-- Confusion Matrix and Statistics -->

<!--           Reference -->
<!-- Prediction     0     1 -->
<!--          0 42609    61 -->
<!--          1    11    40 -->

<!--                Accuracy : 0.998          -->
<!--                  95% CI : (0.998, 0.999) -->
<!--     No Information Rate : 0.998          -->
<!--     P-Value [Acc > NIR] : 0.00147        -->

<!--                   Kappa : 0.526          -->
<!--  Mcnemar's Test P-Value : 7.71e-09       -->

<!--             Sensitivity : 0.396040       -->
<!--             Specificity : 0.999742       -->
<!--          Pos Pred Value : 0.784314       -->
<!--          Neg Pred Value : 0.998570       -->
<!--              Prevalence : 0.002364       -->
<!--          Detection Rate : 0.000936       -->
<!--    Detection Prevalence : 0.001194       -->
<!--       Balanced Accuracy : 0.697891       -->

<!--        'Positive' Class : 1              -->
<!-- ``` -->

<!-- #### 6.2.4 Random Forest -->


<!-- #### 6.2.5 Boosted Trees -->


<!-- #### 6.2.6 SVM -->
