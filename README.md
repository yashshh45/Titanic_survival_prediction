# Titanic Survival Prediction

This repository contains code for predicting survival on the Titanic dataset using various machine learning algorithms. The dataset is loaded using Python libraries like NumPy, Pandas, Seaborn, and Matplotlib. The analysis includes data visualization, data preprocessing, feature engineering, and model training. The models used for prediction are:

1. Logistic Regression
2. K Nearest Neighbor (KNN)
3. Support Vector Machine (SVM) - Linear Classifier
4. Support Vector Machine (SVM) - RBF Classifier
5. Gaussian Naive Bayes
6. Decision Tree Classifier
7. Random Forest Classifier

## Dataset
The Titanic dataset is loaded using Seaborn, containing information about passengers aboard the Titanic, including whether they survived or not. The dataset is preprocessed by dropping unnecessary columns and removing rows with missing values.

## Data Visualization
Several data visualizations are performed using Seaborn and Matplotlib to gain insights into the data and understand the relationships between different features and survival.

## Data Preprocessing
The data is preprocessed to handle missing values and encode categorical features (such as 'sex' and 'embarked') into numerical form using LabelEncoder from Scikit-learn.

## Model Training
The machine learning models are trained on the preprocessed data. The following models are trained:

1. Logistic Regression
2. K Nearest Neighbor (KNN)
3. Support Vector Machine (SVM) - Linear Classifier
4. Support Vector Machine (SVM) - RBF Classifier
5. Gaussian Naive Bayes
6. Decision Tree Classifier
7. Random Forest Classifier

## Model Evaluation
The accuracy of each model is evaluated using confusion matrices and testing accuracy on the test data.

## Feature Importance
The feature importance is calculated for the Random Forest Classifier model, and a bar plot is created to visualize the importance of each feature in predicting survival.

## Making Predictions
A sample data for survival prediction is provided. You can enter your own data in the `my_survival` variable to see if you would have survived on the Titanic based on the trained Random Forest Classifier model.

## Result
The final result will be printed, indicating whether you would have survived or not based on the prediction.

Please note that this is just a demonstration of how to approach a survival prediction problem using machine learning algorithms. The accuracy and performance of the models may vary based on the dataset and feature engineering techniques used.

Feel free to explore the code and dataset to learn more about the Titanic Survival Prediction project. Enjoy!
