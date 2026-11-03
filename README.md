# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create a binary classification dataset.
2. Initialize the weights and bias with zero values.
3. Calculate the predicted probabilities using the sigmoid function and update the weights and bias using gradient descent.
4. Predict the output classes and display the model's accuracy.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S Yeshwanth
RegisterNumber: 212225060310



import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, random_state=42
)

X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.1
epochs = 1000
n = len(y_train)

for i in range(epochs):
    z = np.dot(X_train, weights) + bias
    predictions = 1 / (1 + np.exp(-z))

    dw = np.dot(X_train.T, predictions - y_train) / n
    db = np.sum(predictions - y_train) / n

    weights -= learning_rate * dw
    bias -= learning_rate * db

test_prob = 1 / (1 + np.exp(-(np.dot(X_test, weights) + bias)))
y_pred = (test_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression using Gradient Descent")
print("Actual Values:    ", y_test)
print("Predicted Values: ", y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")
*/
```

## Output:
<img width="931" height="147" alt="image" src="https://github.com/user-attachments/assets/f2deeae9-423e-4ecf-a609-f308d6d4ab0e" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

