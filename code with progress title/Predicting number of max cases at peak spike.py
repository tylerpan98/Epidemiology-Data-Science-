#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn import model_selection


# In[2]:


#read in state data
df=pd.read_csv("Desktop/AllDataCountyfinalfinalfinal.csv")


# In[3]:


#create split using group shuffle to keep state together
gss=GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)

#column to group by
split = gss.split(df, groups=df['county_group'])

#splitting testing and training indexes
train_inds, test_inds = next(split)

#test and train dfs split by state
train=df.iloc[train_inds].copy()
test=df.iloc[test_inds].copy()


# In[8]:


#create y test & train data
y_test = np.array(test['max_cases_per_week'])
y_train = np.array(train['max_cases_per_week'])

#create x test and x train data
x_test = test.iloc[:,-34:]
x_train = train.iloc[:,-34:]


# In[9]:


#baseline model
regtree = DecisionTreeRegressor(random_state=0)
regtree.fit(x_train, y_train)
tree_preds = regtree.predict(x_test)


# In[10]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, tree_preds))
print("MSE:", mean_squared_error(y_test, tree_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, tree_preds)))
print("Score", regtree.score(x_test, y_test))


# In[12]:


#Extra Tree Regression
xtree = ExtraTreeRegressor(random_state=0)
xtree.fit(x_train, y_train)
xtree_preds = xtree.predict(x_test)


# In[13]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xtree_preds))
print("MSE:", mean_squared_error(y_test, xtree_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xtree_preds)))
print("Score", xtree.score(x_test, y_test))


# In[14]:


#random forest
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_preds=rf.predict(x_test)


# In[15]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("MSE:", mean_squared_error(y_test, rf_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, rf_preds)))
print("Score", rf.score(x_test, y_test))


# In[19]:


#feature importance for rf model
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
plt.show()


# In[20]:


#XGB
xgb = XGBRegressor()
xgb.fit(x_train, y_train)
xgb_preds=xgb.predict(x_test)


# In[21]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xgb_preds))
print("MSE:", mean_squared_error(y_test, xgb_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xgb_preds)))
print("Score", xgb.score(x_test, y_test))


# In[22]:


#feature importance for xgb tuned model
importances = xgb.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
plt.show()


# In[23]:


#SVR
scale = MinMaxScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.fit_transform(x_test)


# In[24]:


svr=SVR()
svr.fit(x_train_scaled, y_train)
svr_preds = svr.predict(x_test_scaled)


# In[27]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_preds))
print("MSE:", mean_squared_error(y_test, svr_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_preds)))
print("Score", svr.score(x_test_scaled, y_test))


# In[28]:


#read in state data
df=pd.read_csv("Downloads/AllDataStateFinalFinalFinal.csv")


# In[29]:


#create split using group shuffle to keep state together
gss=GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)

#column to group by
split = gss.split(df, groups=df['state_group'])

#splitting testing and training indexes
train_inds, test_inds = next(split)

#test and train dfs split by state
train=df.iloc[train_inds].copy()
test=df.iloc[test_inds].copy()


# In[31]:


#create y test & train data
y_test = np.array(test['max_var_cases_per_week'])
y_train = np.array(train['max_var_cases_per_week'])

#create x test and x train data
x_test = test.iloc[:,-42:]
x_train = train.iloc[:,-42:]


# In[32]:


#baseline model
regtree = DecisionTreeRegressor(random_state=0)
regtree.fit(x_train, y_train)
tree_preds = regtree.predict(x_test)


# In[33]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, tree_preds))
print("MSE:", mean_squared_error(y_test, tree_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, tree_preds)))
print("Score", regtree.score(x_test, y_test))


# In[34]:


#extra regression trees
xtree = ExtraTreeRegressor(random_state=0)
xtree.fit(x_train, y_train)
xtree_preds = xtree.predict(x_test)


# In[35]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xtree_preds))
print("MSE:", mean_squared_error(y_test, xtree_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xtree_preds)))
print("Score", xtree.score(x_test, y_test))


# In[36]:


#Random Forest
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_preds=rf.predict(x_test)


# In[37]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("MSE:", mean_squared_error(y_test, rf_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, rf_preds)))
print("Score", rf.score(x_test, y_test))


# In[38]:


#feature importance for rf model
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
plt.show()


# In[39]:


#XGB
xgb = XGBRegressor()
xgb.fit(x_train, y_train)
xgb_preds=xgb.predict(x_test)


# In[40]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xgb_preds))
print("MSE:", mean_squared_error(y_test, xgb_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xgb_preds)))
print("Score", xgb.score(x_test, y_test))


# In[41]:


#feature importance for xgb tuned model
importances = xgb.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
plt.show()


# In[42]:


#SVR
scale = MinMaxScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.fit_transform(x_test)


# In[43]:


svr=SVR()
svr.fit(x_train_scaled, y_train)
svr_preds = svr.predict(x_test_scaled)


# In[44]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_preds))
print("MSE:", mean_squared_error(y_test, svr_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_preds)))
print("Score", svr.score(x_test_scaled, y_test))


# In[ ]:




