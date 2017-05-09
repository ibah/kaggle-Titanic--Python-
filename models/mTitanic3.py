# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:40:43 2017
@author: msiwek
Kaggle - Titanic competition

Topic: fast summary of best ideas to build a running model

Models: RandomForest, XGBoost
Ensembling: none -> XGBoost?
Tuning: GridSearch CV
CV: ?

Preprocessing: ?

Conclusions: ?

Plan: ?

"""

print(__doc__)

import numpy.random as npr
from string import ascii_lowercase
from string import digits
source = [x for x in ascii_lowercase+digits]
size = npr.randint(4,6); print(''.join(npr.choice(source, size)))




import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt


# I usually work interactively in Spyder on different platforms.
# The code below allows me to set the proper working directory for each platform.
import os
os.chdir(r'/home/michal/Dropbox/cooperation/_python/Titanic/models')
os.chdir(r'D:\data\Dropbox\cooperation\_python\Titanic\models')
os.chdir(r'G:\Dropbox\cooperation\_python\Titanic\models')

# Loading data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Preprocessing
train.info()
train = train.drop('PassengerId', axis=1)

# Survived
train.Survived.value_counts(normalize=True)
sns.countplot(train.Survived)
plt.hist(train.Survived)
plt.bar([0,1],train.Survived.value_counts(), color=['blue','green'])

# Pclass
train.groupby('Pclass').Survived.mean()
sns.countplot(train.Pclass, hue=train.Survived)

# Name
# Title
train['Name_Title'] = train.Name.apply(lambda x: re.sub('(.*, )|(\..*)','',x))
train.Name_Title.value_counts()
# -> Master (young boy), Mlle, Mme, Ms, Sir, Lady -> Mr or Mrs
train.groupby('Name_Title').Survived.mean().sort_values() # Series
# -> as noted, the survival rates are away from the average level
# Length
train['Name_Length'] = train.Name.apply(lambda x: len(x))
train.groupby('Name_Length').Survived.mean().plot() # there's some relationship
# let's aggregate into bins (5)
pd.qcut(train.Name_Length, 5).value_counts()
# -> data divided into equal quantiles
train.groupby(pd.qcut(train.Name_Length, 5)).size()
train.groupby(pd.qcut(train.Name_Length, 5)).Survived.mean() # Series
train.groupby(pd.qcut(train.Name_Length, 5)).Survived.mean().plot()
# -> see the clear upward trend

# Sex
train['Sex'].value_counts(normalize=True)
sns.countplot(train.Sex, hue=train.Survived)
train.groupby('Sex').Survived.mean() # big difference

# Age
train.groupby(train['Age'].isnull()).Survived.mean()
# -> null:29%, notnull:41%
# check Age impact
plt.plot(train.groupby('Age').Survived.mean()) # too messy
# put Age into bins
train.groupby(pd.qcut(train['Age'], 5)).size() # nice
train.groupby(pd.qcut(train['Age'], 5)).Survived.mean()
train.groupby(pd.qcut(train['Age'], 5)).Survived.mean().plot()

# SibSp
train['SibSp'].value_counts() # actually only 0, 1 have much data; 2,3,4 have some data
train.groupby('SibSp').size() # ditto
train.groupby('SibSp').Survived.mean()
sns.countplot(train.SibSp, hue=train.Survived) # 1-2 seems best

# Parch
train['Parch'].value_counts() # only 0,1,2 have enough data
train.groupby('Parch').Survived.mean()
sns.countplot(train.Parch, hue=train.Survived) # 1-2 seems best, again
np.corrcoef(train.SibSp, train.Parch)[0,1] # there is some positive correlation

# SibSp+Parch
sns.countplot(train.SibSp + train.Parch, hue=train.Survived)
# -> 1-2-3 seem best

# Ticket
train.Ticket[:10] # head(10)
train['Ticket_Len'] = train.Ticket.apply(lambda x: len(x))
train.Ticket_Len.value_counts()
sns.countplot(train.Ticket_Len, hue=train.Survived) # doesn't look very interesting
train['Ticket_Lett'] = train.Ticket.apply(lambda x: x[0]) # str(x)[0]
train.Ticket_Lett.value_counts()
sns.countplot(train.Ticket_Lett, hue=train.Survived) # seems to have some interesting information
train.groupby(['Ticket_Lett']).Survived.mean()

# Fare
train.Fare.plot()
train.Survived.groupby(pd.qcut(train.Fare,5)).mean()
pd.crosstab(pd.qcut(train.Fare,5),train.Pclass)
# -> but remember there are group tickets
#   and fare is a total price for the whole group

# Cabin
# many nulls (700)
train.Cabin[:10]
train.Cabin.dropna()[:10]
# cabin letter
train['Cabin_Letter'] = train.Cabin.apply(lambda x: str(x)[0]) # str needed as there are nans (floats)
train.Cabin_Letter[:10] # n is for 'nan'
train.groupby('Cabin_Letter').Survived.mean()
sns.countplot(train.Cabin_Letter, hue=train.Survived) # intersting
# -> interesting, again NA isn't random
# cabin number
# a high surival rate compared to the population average
train.Cabin.dropna()[:10].apply(lambda x: str(x).split(' ')) # lists of cabin IDs
train.Cabin.dropna()[:10].apply(lambda x: str(x).split(' ')[-1]) # the last cabin ID on each list
train.Cabin.dropna()[:10].apply(lambda x: str(x).split(' ')[-1][1:]) # the last ID withoug the letter
train['Cabin_num'] = train.Cabin.apply(lambda x: str(x).split(' ')[-1][1:])
train.Cabin_num[:10] # an is for nan
train.Cabin_num.replace('an', np.NaN, inplace=True)
train.Cabin_num[:10] # OK
train['Cabin_num'] = train.Cabin_num.apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
train.Cabin_num[:10] # now numbers are as numbers
tmp = pd.qcut(train.Cabin_num, 3)
type(tmp) # Series
tmp.value_counts()
sns.countplot(tmp, hue=train.Survived) # very similar, possibly random? compare with Cabin letter
train.groupby(tmp).Survived.mean()
# -> the very fact the number was recorded makes it more
#    probable that the person has survived.
# this is stupid, but it's there:
train.Survived.corr(train.Cabin_num) # almost null

# Embarked
sns.countplot(train.Embarked, hue=train.Survived)
# -> Survival rate: C > Q ~ S
train.Embarked.value_counts(normalize=True)
train.groupby('Embarked').Survived.mean()
sns.countplot(train.Embarked, hue=train.Pclass)
# -> C has many 1 st class
#    Q is 94% 3rd class, but has a relatively good survival rate
pd.crosstab(train.Embarked, train.Pclass, normalize='index')
# 51% of C is 1st class

# Feature Engineering
# We will perform our feature engineering through a series of helper functions
# that each serve a specific purpose


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
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(str) # lambda not needed
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        # this is some unclear trick, the tickets are grouped:
        # leave as-is: '1', '2', '3', 'S', 'P', 'C', 'A'
        #   (counts 29 or more observations in the train set)
        # set to 'low ticket': 'W', '4', '7', '6', 'L', '5', '8'
        # set to 'other ticket': all other.
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
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

# data preprocessing

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
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
# number of columns
print(len(train.columns))
train.shape[1]

# Hyperparameter Tuning
''' We will use grid search to identify the optimal parameters of our random
 forest model. Because our training dataset is quite small, we can get away
 with testing a wider range of hyperparameter values. When I ran this on my
 8 GB Windows machine, the process took less than ten minutes. I will not run
 it here for the sake of saving myself time, but I will discuss the results of
 this grid search.
 http://scikit-learn.org/stable/modules/grid_search.html
 http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
 http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
'''

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features='auto', # max_features=sqrt(n_features)., default
                            oob_score=True, # use out-of-bag samples to estimate the generalization accuracy
                            random_state=1, # the seed used by the random number generator
                            n_jobs=-1) # the number of jobs is set to the number of cores
param_grid = {"criterion" : ["gini", "entropy"], # function to measure the quality of a split (both options)
              "min_samples_leaf" : [1, 5, 10], # The minimum number of samples required to be at a leaf node
              "min_samples_split" : [2, 4, 10, 12, 16], # The minimum number of samples required to split an internal node
              "n_estimators": [50, 100, 400, 700, 1000]} # The number of trees in the forest.
gs = GridSearchCV(estimator=rf,
                  param_grid=param_grid, # Dictionary with parameters names (string) as keys and lists of parameter settings to try as values
                  scoring='accuracy',
                  cv=3, # use the default 3-fold, default
                  n_jobs=-1)
gs = gs.fit(train.iloc[:, 1:], # X, y
            train.iloc[:, 0]) # office: 4 min, home: 3 min

# saving the gs object into a file, and retriving it
#import os
import pickle
#os.chdir('./17.02Titanic')
#os.getcwd()
with open('gs.pkl','wb') as file:
    pickle.dump(gs,file)
with open('gs.pkl','rb') as file:
    gs_new = pickle.load(file)



'''
GridSearchCV implements a “fit” and a “score” method. It also implements
“predict”, “predict_proba”, “decision_function”, “transform” and
“inverse_transform” if they are implemented in the estimator used.
'''
gs.best_score_ # 0.838
gs.best_params_
'''{'criterion': 'gini',
 'min_samples_leaf': 1,
 'min_samples_split': 10,
 'n_estimators': 700}'''
# -> min sample leaf is very low, suggest possible overfitting... strange
gs.cv_results_
# -> it searched all possible combinations of parameters so:
#    2 x 3 x 5 x 5 = 150 runs
type(gs.cv_results_)
tmp = gs.cv_results_['rank_test_score']
tmp.shape # exactly 150


# Model Estimation and Evaluation

from sklearn.ensemble import RandomForestClassifier

# now we fit the model on the same data with the parameters
# selected by the grid search in the previous step
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train.iloc[:, 1:], train.iloc[:, 0]) # X, y
print("%.4f" % rf.oob_score_) # 0.8294

# variable importance
import pandas as pd
# checking
pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']) # columns in the order as in train
pd.DataFrame(rf.feature_importances_, columns = ['importance']) # just numbers, but the order is the same as in train
pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1)
# -> nice data frame, now all you need is to sort by importance
#    and show the top results
pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
# -> Top variables:
#   sex, Mr, Fare, Name Len, Age, Pclass3

# predictions

predictions = rf.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions.info()
# now add the PassengerID column
# it would be better to preserve it somewhere than to read again from file
os.getcwd()
os.chdir(r'D:\data\Dropbox\cooperation\_python\Titanic')
os.chdir(r'G:\Dropbox\cooperation\_python\Titanic')
os.getcwd()
#test_tmp = pd.read_csv(os.path.join('../input', 'test.csv'))
test_tmp = pd.read_csv(os.path.join('csv','test.csv'))
predictions = pd.concat((test_tmp.iloc[:, 0], predictions),axis = 1)
predictions.to_csv('y_test15.csv', sep=",", index = False)
# Your submission scored 0.82297
# best entry...


# thoughts
"""
I think it's better to stack the two dataframes and then preprocess the whole set.
This is due to the fact that each set has some special values or missing values
and you can't create consistent dummies etc. from just one of them.

I believe I can do better with Fare (group Fares - see my preprocessing)
and Age (better imputation - see Megan).
Interesting simple solutions for Ticket letter and Cabin letter.
Look at family size, maybe you can do the same (some grouping)
for ticket groups and surname groups?
Compare in R random forest, do you use Dummies as here? Check model output.
I like the grid search very much. Simple brutal force tool, but delivers
good result here. You should use it. Maybe look at other methods for
hyperparameter tunning.
The model itself is just a random forest, parameters arrived at by
a grid search.
"""


# trash #######################################################################

# checking dummies generation

train.info()
pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        # some columns are ints - this is only Pclass
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test
# checking
train.info()
# some columns are ints - this is only Pclass
column = 'Pclass'
train[column].head()
train[column] = train[column].apply(str)
train[column].head() # now a string
test[column] = test[column].apply(str)
good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
good_cols # only dummies that are present in the TEST set and TRAIN
[i for i in train[column].unique() if i in test[column].unique()]
np.append(train[column].unique(),test[column].unique())
[i for i in np.unique(np.append(train[column].unique(),test[column].unique()))]
good_cols2 = [column+'_'+i for i in
              np.unique(np.append(train[column].unique(),test[column].unique()))]
good_cols2 # dummies in TRAIN and/or in TEST sets
pd.get_dummies(train[column], prefix=column).head()
pd.get_dummies(train[column], prefix=column).head()[good_cols] # only cols present in both sets
# now add these new dummy columns to the sets
train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
# now delete the original columns
del train[column]
del test[column]


# droping wrong columns
predictions.drop('Age',1,inplace=True)
predictions.info()

# trash

x = train.Cabin[27]
x
int(re.search('[0-9]*$',x).group())
pd.isnull(train.Cabin[0])
y = train.Cabin.dropna().apply(lambda x: re.search('[0-9]*$',str(x)).group())
y[:10]
y.apply(int) # there are some values that are not numbers
row=-1
for i in y:
    row += 1
    try:
        int(i)
    except:
        print('The source of error:['+i+'] in row '+str(row))
y[[59,72,78,105]]
y = train.Cabin.apply(lambda x: 777 if pd.isnull(x) else print(x), int(re.search('[0-9]*$',str(x)).group()))
