# Diabetes_Prediction-KNN

## Introduction
This project utilizes the K-Nearest Neighbors (KNN) algorithm to predict diabetes based on various medical attributes. The objective is to build a model that can accurately classify whether a patient has diabetes or not based on specific health metrics.  

## About the Dataset
The dataset used in this project includes the following features:  
- **Pregnancies**: Number of times pregnant.  
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.  
- **BloodPressure**: Diastolic blood pressure (mm Hg).  
- **SkinThickness**: Triceps skin fold thickness (mm).  
- **Insulin**: 2-Hour serum insulin (mu U/ml).  
- **BMI**: Body mass index (weight in kg/(height in m)^2).  
- **DiabetesPedigreeFunction**: Diabetes pedigree function.  
- **Age**: Age (years).  
- **Outcome**: Class variable (0 or 1), indicating whether the patient has diabetes (1) or not (0).  

## Tools and Libraries
- **Python**: Programming language used for data processing and model building.  
- **Pandas**: For data manipulation and analysis.  
- **NumPy**: For numerical operations.  
- **Scikit-learn**: For implementing the KNN algorithm and evaluating the model.  
- **Seaborn**: For enhanced data visualization.  

## Steps
1. **Data Loading and Preprocessing**:  
   - Load the dataset from a CSV file.  
   - Handle missing values if any.  
   - Standardize the features to ensure the KNN algorithm works effectively.  

2. **KNN Model Training**:  
   - Split the dataset into training and testing sets.  
   - Train the KNN model using the training data.  
   - Optimize the number of neighbors (K) using cross-validation.  

3. **Model Evaluation**:
   - Evaluate the model on the test data using accuracy, precision, recall, and F1-score.  
   - Generate a confusion matrix to visualize the performance.  

## Outcome
The project successfully builds and evaluates a KNN model to predict diabetes based on the provided features. The model's performance is assessed using various metrics to ensure its accuracy and reliability.  

## Conclusion
The KNN algorithm is effective for predicting diabetes based on medical attributes. The model can be further improved by tuning hyperparameters and incorporating additional features or advanced techniques.  

## Metrics
- **Accuracy**: 75%  
- **F1-score**: 62%  
