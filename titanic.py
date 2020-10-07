# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 14:04:13 2020

@author: Harsh Garg
"""

#import relevant libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import mode

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#import the training and testing dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#basic data exploration
train.describe()
train.info()

women = train[train.Sex=='female']
len(women[women.Survived==1])/len(women) #percentage of women who survived

men = train[train.Sex=='male']
len(men[men.Survived==1])/len(men) #percentage of men who survived

train.columns #gives all column names

y = train['Survived'] #the ground truth
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'] #possible relevant features
X = pd.get_dummies(train[features]) 
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)

#training on various models

#first random forest classifier
rf = RandomForestClassifier()

# default random forest accuracy check
rf.fit(X_train, y_train)
rf_y = rf.predict(X_test)
print(accuracy_score(y_test, rf_y)) #0.7988826815642458

# now we tune hyperparameters using random search CV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(10, 150, num = 15)]
# Criteria to use
max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth}
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 150, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random_y = rf_random.best_estimator_.predict(X_test)
print(accuracy_score(y_test, rf_random_y)) #0.8100558659217877
#print(rf_random.best_params_)

#Second is Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_y = nb.predict(X_test)
print(accuracy_score(y_test, nb_y)) #0.776536312849162

#Third is KNeighbours
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
kn_y = kn.predict(X_test)
print(accuracy_score(y_test, kn_y)) #0.7821229050279329

# now we tune hyperparameters using random search CV
# Number of neighbours
neighbours = [int(x) for x in np.linspace(2,10, num = 9)]
# Weights used
weights = ['uniform', 'distance']
#Power parameter
p = [int(x) for x in np.linspace(1,5, num = 5)]
random_grid = {'n_neighbors': neighbours,
               'weights': weights,
               'p': p}
kn_random = RandomizedSearchCV(estimator = kn, param_distributions = random_grid, 
                               n_iter = 90, cv = 3, verbose=2, random_state=42, n_jobs = -1)
kn_random.fit(X_train, y_train)
kn_random_y = kn_random.best_estimator_.predict(X_test)
print(accuracy_score(y_test, kn_random_y)) #0.7932960893854749

#Ensemble of the above 3 methods used

ensemble_random_y = []
for x in range(len(X_test)):
    ensemble_random_y.append(mode([rf_random_y[x], nb_y[x], kn_random_y[x]]))

print(accuracy_score(y_test, ensemble_random_y)) #0.7877094972067039

ensemble_y = []
for x in range(len(X_test)):
    ensemble_y.append(mode([rf_y[x], nb_y[x], kn_y[x]]))

print(accuracy_score(y_test, ensemble_y)) #0.7877094972067039

