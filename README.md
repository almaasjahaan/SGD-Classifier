# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.
2.Data preprocessing:
3.Cleanse data,handle missing values,encode categorical variables.
4.Model Training:Fit logistic regression model on preprocessed data.
5.Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
6.Prediction: Predict placement status for new student data using trained model.
7.End the program 


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: M ALMAAS JAHAAN
RegisterNumber:  212224230016
*/

import pandas as pd

# Load the dataset
data = pd.read_csv("C:\\Users\\admin\\Downloads\\Employee.csv")
data.head()
data.info()
print("Null values:\n", data.isnull().sum())
print("Class distribution:\n", data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder

# Encode categorical features
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Display the updated data
data.head()

# Select input features
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
print(x.head())

# Define target variable
y = data["left"]

from sklearn.model_selection import train_test_split

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier

# Create and train the model
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

from sklearn import metrics

# Predict on test set
y_pred = dt.predict(x_test)

# Evaluate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on new employee data
sample_prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Sample Prediction:", sample_prediction)

```

## Output:
![prediction of iris species using SGD Classifier](sam.png)

![image](https://github.com/user-attachments/assets/73c37b1c-2d87-4882-9549-42ddc3330662)

![image](https://github.com/user-attachments/assets/e30c3015-dfed-4e1f-96be-38109ddc49ee)

![image](https://github.com/user-attachments/assets/9f6f6ff2-fe6f-4377-9ea4-c3d0e03e2c4d)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
