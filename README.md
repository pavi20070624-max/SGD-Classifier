# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.)Import the required libraries such as Pandas, Matplotlib, Seaborn, and Scikit-learn modules.

2.)Load the Iris dataset and create a DataFrame containing feature values and target labels.

3.)Separate the dataset into input features (X) and target variable (y).

4.)Split the dataset into training and testing sets using train-test split.

5.)Create the SGD Classifier model with suitable parameters like maximum iterations and tolerance.

6.)Train the model using the training dataset and predict iris species for the test dataset.

7.)Evaluate the model performance by calculating accuracy score and generating the confusion matrix.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: PAVITHRA S
RegisterNumber:  212225040298

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(cm)

```

## Output:
<img width="862" height="322" alt="image" src="https://github.com/user-attachments/assets/82a8504f-ea2c-4a5f-a545-c7eae99b84ae" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
