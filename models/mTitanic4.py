#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:03:06 2017
@author: michal
Kaggle - Titanic competition

Topic: fast summary of best ideas to build a running model

Models: RandomForest
Ensembling: none -> XGBoost?
Tuning: GridSearch CV
CV: default 3-fold
Inspriation: https://www.kaggle.com/zlatankr/titanic-random-forest-82-78 by Zlatan Kremonic

Preprocessing: ?

Conclusions: ?

Plan: ?


http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer
http://scikit-learn.org/stable/auto_examples/preprocessing/plot_function_transformer.html#sphx-glr-auto-examples-preprocessing-plot-function-transformer-py
"""

print(__doc__)

import numpy as np
import pandas as pd
import re

# I work interactively in Spyder on different platforms.
# The code below allows me to set the proper working directory for each platform.
import os
os.chdir(r'/home/michal/Dropbox/cooperation/_python/Titanic/models')
os.chdir(r'D:\data\Dropbox\cooperation\_python\Titanic\models')
os.chdir(r'G:\Dropbox\cooperation\_python\Titanic\models')

# Loading data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Preprocessing
train = train.drop('PassengerId', axis=1)
test_id = test.pop('PassengerId')
train['Name_Title'] = train.Name.apply(lambda x: re.sub('(.*, )|(\..*)','',x))
train['Name_Length'] = train.Name.apply(len)
train['Ticket_Len'] = train.Ticket.apply(len)
train['Ticket_Lett'] = train.Ticket.apply(lambda x: x[0]) # str(x)[0]
train['Cabin_Letter'] = train.Cabin.apply(lambda x: str(x)[0]) # str needed as there are nans (floats)
train['Cabin_num'] = train.Cabin.apply(lambda x: str(x).split(' ')[-1][1:])
train.Cabin_num.replace('an', np.NaN, inplace=True) # an is for nan
train['Cabin_num'] = train.Cabin_num.apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

def names(train, test):
    # Name and Title
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
        del i['Name']
    return train, test
def age_impute(train, test):
    # Age imputation, mean per class & title
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass']).Age
        # OK, here we have a question: should we use the mean
        # taken from the respective train or test set?
        # or should we use means from the train only?
        # In practical machine learning the values used in preprocessing
        # were supposed to come always from the train set only
        # e.g. when normalizing some predictor, you should use
        # values for mu and sigma estimated on the training set
        # - use them both for training and test sets.
        # so let's leave the train here too.
        i['Age'] = data.transform(lambda x: x.fillna(x.mean())) # returns a Series
        # fill NAs with means
    return train, test
def fam_size(train, test):
    # family size
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                            np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear',
                                     'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test
def ticket_grouped(train, test):
    # Ticket lenght, ticket letter
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        #i['Ticket_Lett'] = i['Ticket_Lett'].apply(str) # lambda not needed
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        # this is some unclear trick, the tickets are grouped:
        # leave as-is: '1', '2', '3', 'S', 'P', 'C', 'A'
        #   (counts 29 or more observations in the train set)
        # set to 'low ticket': 'W', '4', '7', '6', 'L', '5', '8'
        # set to 'other ticket': all other.
        i['Ticket_Len'] = i['Ticket'].apply(len)
        del i['Ticket']
    return train, test
def cabin(train, test):
    # cabin letter
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test
def cabin_num(train, test):
    # cabin number
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
        # qcut into 3 ranges
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    # columns are now broken into dummy variables
    # here one should set drop_first=True
    # to remove collinearity between the dummies
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    # the other columns can be deleted
    return train, test
def embarked_impute(train, test):
    # impute Embarked
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test
# impute Fare
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
# -> this code is repeated below.
# -> fare NAs are simply filled in with mean fare
#    this seems to be very crude
#    also what to do with fare = 0?
# Because we are using scikit-learn,
# we must convert our categorical columns into dummy variables
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(str) # lambda not needed
        test[column] = test[column].apply(str) # lambda not needed
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test
def drop(train, test, bye = ['PassengerId']):
    # drop unnecessary columns: PassengerId
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test

# loading data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train = train.drop('PassengerId', axis=1)
train_y = train.pop('Survived')
test_id = test.pop('PassengerId')

# data preprocessing
train, test = names(train, test)
train, test = age_impute(train, test)
train, test = cabin_num(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = fam_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = ticket_grouped(train, test)
train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                              'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = drop(train, test)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features='auto',
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1)
params = {"min_samples_leaf" : [1, 5, 10],
         "min_samples_split" : [2, 4, 10, 12, 16],
         "n_estimators": [50, 100, 400, 700, 1000]}
gs = GridSearchCV(rf, params, n_jobs=-1)
gs.fit(train, train_y) # office: 4 min, home: 3 min
gs.best_score_ # 0.838
gs.best_params_
pred = gs.predict(test)
pred = pd.DataFrame({'PassengerId': test_id, 'Survived': pred})
pred.to_csv('y_test17.csv', index = False)









