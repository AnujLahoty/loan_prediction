# Logistic Regression

# Importing the libraries

import pandas as pd
from sklearn.preprocessing import Imputer

# Importing the dataset

# Dividing into the categeorical attributea
dataset = pd.read_csv('loan.csv')
dataset_cat = dataset.iloc[:, [4, 5, 11, 12]]
dataset_cat = pd.get_dummies(dataset_cat, drop_first = True)
dataset_cat.info()

# Dividing into the numerical attributes
dataset_num = dataset.iloc[:, [6, 7, 8, 9, 10]]

# Taking care of the missing data.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset_num.iloc[:, :])
dataset_num.iloc[:, :] = imputer.transform(dataset_num.iloc[:, :]) 
dataset_num.info()

# seprating the target variable
dataset = pd.concat((dataset_cat, dataset_num), axis = 1)

X = dataset.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].values
y = dataset.iloc[:, [4]].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Assesing the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

