#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR


# In[2]:


#read in county data
df=pd.read_csv("Desktop/AllDataCountyfinalfinalfinal.csv")


# In[8]:


#create split using group shuffle to keep state together
gss=GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)

#column to group by
split = gss.split(df, groups=df['county_group'])

#splitting testing and training indexes
train_inds, test_inds = next(split)

#test and train dfs split by state
train=df.iloc[train_inds].copy()
test=df.iloc[test_inds].copy()


# In[9]:


#create y test & train data
y_test = np.array(test['diff_weeks'])
y_train = np.array(train['diff_weeks'])

#create x test and x train data
x_test = test.iloc[:,-34:]
x_train = train.iloc[:,-34:]

x_test_no_correlated = x_test.drop(['Mutation Fitness', 'Number of business establishments', 'Total people in poverty (%)', 'Population_Density'], axis=1)
x_train_no_correlated = x_train.drop(['Mutation Fitness', 'Number of business establishments', 'Total people in poverty (%)','Population_Density'], axis=1)


# In[10]:


#array of the group column
group_train=np.array(train['county_group'])


# In[5]:


from sklearn.feature_selection import SelectFromModel


# In[13]:


rf=RandomForestRegressor()
sel = SelectFromModel(rf, max_features = 20)
sel.fit(x_train_no_correlated, y_train)


# In[14]:


selected_feat= x_train_no_correlated.columns[(sel.get_support())]
len(selected_feat)


# In[15]:


print(selected_feat)


# In[16]:


x_train_features = x_train_no_correlated[["S1 Mutations", "21A (Delta)", "Number of residents",
       "1st_Quarter_first", "2nd_Quarter_first", "Omicron",
       "% Fair or Poor Health", 'Unempl rate']]


# In[17]:


x_test_feature_scale = x_test_no_correlated[["S1 Mutations", "21A (Delta)", "Number of residents",
       "1st_Quarter_first", "2nd_Quarter_first", "Omicron",
       "% Fair or Poor Health",'Unempl rate']]


# In[20]:


scale = MinMaxScaler()
x_train_feature_scaled = scale.fit_transform(x_train_features)
x_test_feature_scaled = scale.fit_transform(x_test_feature_scale)


# In[23]:


svr = SVR()
svr.fit(x_train_feature_scaled, y_train)
svr_preds = svr.predict(x_test_feature_scaled)


# In[24]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_preds))
print("MSE:", mean_squared_error(y_test, svr_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_preds)))


# In[25]:


make_pipe = make_pipeline(scale, SVR())
#param = {'kernel':['poly','rbf','linear'],'degree':[1,2,3,4,5], 'gamma':[0.1,0.025,0.05,0.01]}
param = {'svr__kernel':['poly','rbf','linear'],'svr__degree':[1,2,3,4,5],'svr__gamma':['scale', 'auto']}
gkf=GroupKFold(n_splits=5).split(x_train_features, y_train,group_train )
gridsvr = GridSearchCV(make_pipe, param_grid = param, scoring = 'neg_root_mean_squared_error', cv=gkf)
result=gridsvr.fit(x_train_features, y_train )
print(gridsvr.best_params_)


# In[26]:


#predicting test data using best parameters from grid search
tuned_pred_svm = gridsvr.predict(x_test_feature_scaled)


# In[27]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, tuned_pred_svm))
print("MSE:", mean_squared_error(y_test, tuned_pred_svm))
print("RMSE:", math.sqrt(mean_squared_error(y_test, tuned_pred_svm)))


# In[28]:


svr = SVR(degree = 5, gamma = 'scale', kernel = 'poly')


# In[29]:


svr.fit(x_train_feature_scaled, y_train)
svr_preds = svr.predict(x_test_feature_scaled)


# In[30]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_preds))
print("MSE:", mean_squared_error(y_test, svr_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_preds)))


# In[32]:


from sklearn import model_selection
#cross validation of results using best parameters on training data
gkf2=GroupKFold(n_splits=10)
pipe = make_pipeline(scale, SVR(degree = 5, gamma = 'scale', kernel = 'poly'))
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(pipe, x_train_feature_scaled, y_train, scoring =scores, cv=gkf2, groups=group_train)
print(scores)


# In[ ]:




