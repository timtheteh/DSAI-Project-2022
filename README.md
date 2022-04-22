# CZ1115-DSAI Predicting Wine Quality
![image](https://user-images.githubusercontent.com/61932721/164416481-03f3dbd9-bfca-403d-96ed-5f9ff28771f4.png)

Spoilt for choice? With such a large variety of wine in the shelves of your local supermarket, how can you make an informed decision to enjoy a lovely drink without becoming a pseudo alcoholic?

This is a mini project for CZ1115 Introduction to Data Science and Artificial Intelligence, which aims to predict the wine quality from the physicochemical properties of wine.

## Problem Definition
* Given the physicochemical properties of a wine, can we predict whether a wine is of good quality (Rating >= 6)?
* Which model best predicts the wine quality?

## Procedure
1\. Exploratory Data Analysis
* Exploratory data analysis of wine dataset
* Removing duplicates in dataset
* Reformatting 'quality' to a binary variable
* Exporting reformatted dataset into .csv file

2.1 Modelling without Duplicates, 2.2 Modelling without Outliers,\
2.3 Modelling with Feature Selection, 2.4 Modelling without Outliers and with Feature Selection
* Comparing performance of models with and without feature selection
* Comparing performance of models with and without outlier removal

3\. Final Modelling with SMOTE
* Integrating SMOTE to reduce sample imbalance in machine learning process
* Hyperparameter tuning using GridSearch with K-Fold cross-validation to optimise model accuracy
* Implementing and comparing 7 different models as stated below
* Deliver insights gathered from data analysis process and model implementations

## Models Used
1. Logistic Regression
2. Gaussian Naive Bayes
3. Decision Tree Classifier
4. Random Forest Classifier
5. AdaBoost Classifier
6. CatBoost Classifier
7. Gradient Boosting Classifier

## Insights from the Project
* Only some attributes of wine have a relationship with wine quality. For instance, more wines with higher alcohol levels are 'good'. On the other hand, the pH values of a wine does not seem to affect whether a wine will be perceivied as 'good' or 'bad'.
* Across all wines, some variables such as chlorides follow a normal distribution. On the other hand, other variables like citric acid levels do not follow a normal distribution.
* Not all data cleaning methods may be appropriate, as feature selection and outlier removal led to lowered model accuracy.
* Random Forest Classifier was the best model to be used in predicting whether a wine is of good quality, with an accuracy of 77.43% on the test dataset.
* Decision Tree Classifier performed the worst, with an accuracy of 67.71% on the test dataset.
* It is possible to predict whether a wine is of good quality, given the physicochemical properties of the wine.

## Contributors
@timtheteh - Data Extraction, Data Cleaning, Data Visualisation

@hashtagyx - Implementation of Machine Learning Models: Logistic Regression, Gaussian Naive Bayes, Decision Tree

@zero0zero7 - Implementation of Machine Learning Models: Random Forest, AdaBoost, CatBoost, Gradient Boosting Classifier


## References
* https://archive.ics.uci.edu/ml/datasets/wine+quality
* https://www.kaggle.com/code/firuzjuraev/red-wine-quality-forecasting/notebook#--Modeling!-
* https://www.analyticsvidhya.com/blog/2021/04/distinguish-between-tree-based-machine-learning-algorithms/
* https://towardsdatascience.com/the-right-way-of-using-smote-with-cross-validation-92a8d09d00c7
* https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub
* https://blog.paperspace.com/adaboost-optimizer/#:~:text=AdaBoost%20is%20an%20ensemble%20learning,turn%20them%20into%20strong%20ones
