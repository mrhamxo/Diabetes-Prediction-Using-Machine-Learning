# Diabetes Prediction Using Machine Learning

## Project Overview
This project focuses on predicting diabetes using various machine learning algorithms based on health data. The dataset includes multiple features such as glucose level, blood pressure, insulin level, BMI, and age. The goal is to build a predictive model that can accurately classify whether a person has diabetes or not.

## Table of Contents
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Algorithms and Accuracy](#algorithms-and-accuracy)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Project Conclusion](#project-conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Dataset
The dataset used in this project is sourced from the Pima Indians Diabetes Database, which is available on Kaggle. It contains 768 observations with 8 features and one target variable, `Outcome`, which indicates the presence of diabetes (1) or absence (0).

## Project Workflow
1. **Data Preprocessing:**
   - Handle missing values and outliers.
   - Perform one-hot encoding on categorical variables (`newBMI`, `NewInsulinScore`, `newGlucose`).
   - Apply feature scaling using `RobustScaler` and `StandardScaler`.

2. **Feature Selection:**
   - Drop certain columns to reduce the total features from 18 to 8, focusing on the most impactful features.

3. **Model Training:**
   - Train several machine learning models using the preprocessed data.
   - Perform hyperparameter tuning where necessary.

4. **Model Evaluation:**
   - Evaluate models based on accuracy and other metrics like precision, recall, and F1-score.

## Algorithms and Accuracy
The following machine learning algorithms were used to predict diabetes, along with their respective accuracy on the test data:

| SNo | Algorithm                    | Accuracy |
|-----|------------------------------|----------|
|  1  | K-Nearest Neighbors (KNN)    |  90.13%  |
|  2  | Random Forest Classifier     |  90.13%  |
|  3  | Support Vector Machine (SVM) |  89.47%  |
|  4  | Logistic Regression          |  88.82%  |
|  5  | Gradient Boosting Classifier |  88.16%  |
|  6  | XGBBoost Classifier          |  88.16%  |
|  7  | Decision Tree Classifie      |  85.53%  |

## Feature Engineering
Feature engineering involved transforming and encoding the following features:
- **newBMI**: Categorized into `Underweight`, `Normal`, `Overweight`, `Obesity 1`, and `Obesity 2`.
- **NewInsulinScore**: Categorized into `Normal`, `Abnormal`.
- **newGlucose**: Categorized into `Low`, `Normal`, `High`, and `Very High`.

After one-hot encoding, less relevant columns were dropped to reduce feature dimensionality.

## Model Training
The data was split into training and testing sets. Several models were trained on the training data, and hyperparameter tuning was performed using grid search and cross-validation to find the best performing model.

## Model Evaluation
The models were evaluated on the test set using accuracy, precision, recall, and F1-score. The Random Forest Classifier achieved the highest accuracy of 81.4%, making it the best-performing model in this project.

## Model Performance and ROC Graphs
![PE_diabetes](https://github.com/user-attachments/assets/5c1df603-8919-4a99-b0c6-338079677e06)

![roc_diabetes](https://github.com/user-attachments/assets/1ce9d596-238c-4c78-b101-5401237bd6a4)

## Project Conclusion
The Random Forest Classifier proved to be the most effective algorithm for predicting diabetes with an accuracy of 81.4%. However, other models like Gradient Boosting Classifier and SVM also showed competitive performance. The feature engineering steps, particularly one-hot encoding and feature scaling, played a crucial role in improving the model's accuracy.

This project demonstrates the importance of data preprocessing and feature engineering in building robust predictive models. The successful prediction of diabetes can aid in early diagnosis and treatment, potentially saving lives.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-ml.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
- The project includes a Streamlit app that allows users to input their health data and get predictions about whether they have diabetes.
- Simply input the required features and click "Predict" to see the results.

## References
- [Pima Indians Diabetes Database on Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
