# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:20:38 2022

@author: User
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher, 
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) 
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% STATICS

CSV_PATH = os.path.join(os.getcwd(),'heart.csv')

# EDA
# Step 1) Data loading

df = pd.read_csv(CSV_PATH)

# Step 2) Data Inspection

df.info() # to check for Null, 
df.describe() 

df.head() # to print first 5 in data

plt.figure(figsize=(20,20))
df.boxplot() 
plt.show()
# from the data from boxplot, trtbps and chol have outliers 

df.duplicated().sum() # Has 1 duplicated data
df[df.duplicated()] # Extract duplicated data

# categorical column
cat = ['sex','cp','fbs','restecg','exng','slp','caa','output','thall']

# continous column
con = ['age','trtbps','chol','thalachh','oldpeak']

# to see the dispersion of data
cater_columns = ['sex','cp','fbs','restecg','exng','slp','caa','output','thall']
for cat in cater_columns:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()
    

# The data consists twice number of male compared to number of female

con_columns = ['age','trtbps','chol','thalachh','oldpeak']
for con in con_columns:
    plt.figure()
    sns.displot(df[con])
    plt.show()



# Step 3) Data Cleaning
# Remove duplicated data
df = df.drop_duplicates() # to remove duplicate data
df.duplicated().sum() # Has 0 duplicated data

# Data imputation using Simple Imputer
df['thall']=df['thall'].replace(0,np.nan) # 0 is replaced with NaNs
df.isna().sum() # shows 2 NaNs
df['thall'].fillna(df['thall'].mode()[0], inplace=True) # NaNs replaced with mode
df.isna().sum() # shows 0 Nans

# NaNs value of thall have been replaced with mode

# Step 4) Feature Selection
# cat = ['sex','cp','fbs','restecg','exng','slp','caa','output','thall']
# con = ['age','trtbps','chol','thalachh','oldpeak']

# continous vs categorical
for con in con_columns:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis=-1),df['output'])
    print(lr.score(np.expand_dims(df[con],axis=-1),df['output']))

# categorical vs categorical
for cat in cater_columns:
    print(cat)
    confussion_mat = pd.crosstab(df[cat],df['output']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))
    
# will be selecting parameter that has accuracy more than 0.5 for continous  
# will be selecting parameter that has correlation more than 0.5 for categorical  
  
# new_X = ['age','trtbps','chol','thalachh','oldpeak','cp','thall']

X = df.loc[:,['age','trtbps','chol','thalachh','oldpeak','cp','thall']]
y = df.loc[:,'output']

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=(123))

#%% pipeline


# Logistics Regression Pipeline
pl_std_lr = Pipeline([('StandardScaler',StandardScaler()),
                      ('LogisticsClassifier',LogisticRegression())])

pl_mms_lr = Pipeline([('MinMaxScaler',MinMaxScaler()),
                      ('LogisticsClassifier',LogisticRegression())])

# KNN Pipeline
pl_std_knn = Pipeline([('StandardScaler',StandardScaler()),
                      ('KNNClassifier',KNeighborsClassifier())])

pl_mms_knn = Pipeline([('MinMaxScaler',MinMaxScaler()),
                      ('KNNClassifier',KNeighborsClassifier())])

# RF Pipeline
pl_std_rf = Pipeline([('StandardScaler',StandardScaler()),
                      ('RFClassifier',RandomForestClassifier())])

pl_mms_knn = Pipeline([('MinMaxScaler',MinMaxScaler()),
                      ('RFClassifier',RandomForestClassifier())])

# DT Pipeline
pl_std_dt = Pipeline([('StandardScaler',StandardScaler()),
                      ('DTClassifier',DecisionTreeClassifier())])

pl_mms_dt = Pipeline([('MinMaxScaler',MinMaxScaler()),
                      ('DTClassifier',DecisionTreeClassifier())])

#SVC Pipeline
pl_std_svc = Pipeline([('StandardScaler',StandardScaler()),
                      ('SVClassifier',SVC())])

pl_mms_svc = Pipeline([('MinMaxScaler',MinMaxScaler()),
                      ('SVCClassifier',SVC())])

# Model Analysis

pipelines = [pl_std_lr,
             pl_mms_lr,
             pl_std_knn,
             pl_mms_knn,
             pl_std_rf,
             pl_mms_knn,
             pl_std_dt,
             pl_mms_dt,
             pl_std_svc,
             pl_mms_svc]

for pipeline in pipelines:
    pipeline.fit(X_train,y_train)
    
best_accuracy = 0
for i, pipeline in enumerate(pipelines):
    print(pipeline.score(X_test,y_test))
    if pipeline.score(X_test,y_test) > best_accuracy:
        best_accuracy = pipeline.score(X_test,y_test)
        best_pipeline = pipeline
print('The best combination of the pipeline is {} with accuracy of {}'.format(best_pipeline.steps,best_accuracy))

from sklearn.model_selection import GridSearchCV

# define model and parameters
grid_param = [{'LogisticsClassifier__solver':['lbfgs','newton-cg','liblinear'],
               'LogisticsClassifier__penalty':['l2'],
               'LogisticsClassifier__C':[100,10,1.0,0.1,0.01]}]

grid_search = GridSearchCV(pl_mms_lr,grid_param,cv=5,verbose=0,n_jobs=-1)
best_model = grid_search.fit(X_train,y_train)

print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

#%% Retrain the model with the selected parameters

pl_mms_lr = Pipeline([('MinMaxScaler',MinMaxScaler()),
                      ('LogisticsClassifier',LogisticRegression(solver='lbfgs',
                                                                penalty='l2',
                                                                C=1.0))])
                          
pl_mms_lr.fit(X_train,y_train)

BEST_PIPE_MODEL =  os.path.join(os.getcwd(),'best_model.pkl')
with open(BEST_PIPE_MODEL,'wb') as file:
    pickle.dump(pl_mms_lr,file)

#%% Pickle file

BEST_PIPE_PATH =  os.path.join(os.getcwd(),'best_pipeline.pkl')
with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

#%% MODEL ANALYSIS

y_true = y_test
y_pred = best_model.predict(X_test)

print(classification_report,y_true,y_pred)
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

#%% Discussion

# Male have more chance to get heart attack than female
# MMS + LogisticRegression is the best pipeline model
