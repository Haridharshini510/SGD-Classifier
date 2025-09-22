# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data.

2.Split Dataset into Training and Testing Sets.

3.Train the Model Using Stochastic Gradient Descent (SGD).

4.Make Predictions and Evaluate Accuracy.

5.Generate Confusion Matrix. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Haridharshini J
RegisterNumber: 212224040098 
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SGD Classifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

# Predictions
y_pred = sgd_clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
![prediction of iris species using SGD Classifier](sam.png)
<img width="889" height="166" alt="image" src="https://github.com/user-attachments/assets/3c8f3469-5369-4fdc-a2a3-e3a52653c45c" />

<img width="230" height="306" alt="image" src="https://github.com/user-attachments/assets/e618254d-8f6d-4a10-8817-0e0d73bb935d" />

<img width="738" height="544" alt="image" src="https://github.com/user-attachments/assets/8961fbea-54c3-4f34-9635-0c99eb0863d9" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
