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
df=pd.read_csv("Downloads/AllDataStateFinalFinalFinal.csv")


# In[3]:


df.head()


# In[6]:


features = df.iloc[:,[13,14,15,25, 26,27,28,29,30,31,32,33,34,35,36,37,38,39,50,51,52,53,54,55 ]]
features.head()


# In[7]:



correlations = features.corr()

fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .80})
plt.savefig('Downloads/county_corr_matrix.png')
plt.show();
    

    


# In[9]:


#create split using group shuffle to keep state together
gss=GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)

#column to group by
split = gss.split(df, groups=df['state_group'])

#splitting testing and training indexes
train_inds, test_inds = next(split)

#test and train dfs split by state
train=df.iloc[train_inds].copy()
test=df.iloc[test_inds].copy()


# In[12]:


#create y test & train data
y_test = np.array(test['diff_weeks'])
y_train = np.array(train['diff_weeks'])

#create x test and x train data
x_test = test.iloc[:,-42:]
x_train = train.iloc[:,-42:]

x_test_no_corr = x_test.drop(['Mutation Fitness', '#_Public_Airports' ,'#_business_establishments','Miles freight railroad','Miles passenger railroad' ,'PovertyRate'], axis=1)
x_train_no_corr = x_train.drop(['Mutation Fitness', '#_Public_Airports' , '#_business_establishments','Miles freight railroad','Miles passenger railroad' ,'PovertyRate'], axis=1)


# In[14]:


#array of the group column
group_train=np.array(train['state_group'])


# In[15]:


#baseline model
regtree = DecisionTreeRegressor(random_state=0)
regtree.fit(x_train_no_corr, y_train)
tree_preds = regtree.predict(x_test_no_corr)


# In[17]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, tree_preds))
print("MSE:", mean_squared_error(y_test, tree_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, tree_preds)))
print("Score", regtree.score(x_test_no_corr, y_test))


# In[18]:


#cross validation of metrics using training data
gkf2=GroupKFold(n_splits=10)
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(regtree, x_train_no_corr, y_train, scoring = scores, cv=gkf2, groups=group_train)
print(scores)


# In[19]:


#ensemble modeling Extra Regressor Tree 


# In[20]:


xtree = ExtraTreeRegressor(random_state=0)
xtree.fit(x_train_no_corr, y_train)
xtree_preds = xtree.predict(x_test_no_corr)


# In[21]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xtree_preds))
print("MSE:", mean_squared_error(y_test, xtree_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xtree_preds)))
print("Score", xtree.score(x_test_no_corr, y_test))


# In[29]:


xtree_tuned = ExtraTreeRegressor(max_leaf_nodes= 30, min_samples_split= 5, splitter= 'random')
xtree_tuned.fit(x_train_no_corr, y_train)
xtree_tuned_preds = xtree_tuned.predict(x_test_no_corr)


# In[30]:


#cross validation of metrics using training data
gkf2=GroupKFold(n_splits=10)
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(xtree_tuned, x_train_no_corr, y_train, scoring = scores, cv=gkf2, groups=group_train)
print(scores)


# In[31]:


#evaluation metrics of model on test data
print("MAE:", mean_absolute_error(y_test, xtree_tuned_preds))
print("MSE:", mean_squared_error(y_test, xtree_tuned_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xtree_tuned_preds)))
print("Score", xtree_tuned.score(x_test_no_corr, y_test))


# In[32]:


#random forest


# In[33]:


rf = RandomForestRegressor()
rf.fit(x_train_no_corr, y_train)
rf_preds=rf.predict(x_test_no_corr)


# In[34]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("MSE:", mean_squared_error(y_test, rf_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, rf_preds)))
print("Score", rf.score(x_test_no_corr, y_test))


# In[38]:


rf_tuned=RandomForestRegressor(max_depth = 30, n_estimators= 30)
rf_tuned.fit(x_train_no_corr, y_train)
rf_preds_tuned = rf_tuned.predict(x_test_no_corr)


# In[39]:


#cross validation of metrics using training data
gkf2=GroupKFold(n_splits=10)
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(rf_tuned, x_train_no_corr, y_train, scoring = scores, cv=gkf2, groups=group_train)
print(scores)


# In[40]:


#evaluation metrics of model on test data
print("MAE:", mean_absolute_error(y_test, rf_preds_tuned))
print("MSE:", mean_squared_error(y_test, rf_preds_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, rf_preds_tuned)))
print("Score", rf_tuned.score(x_test_no_corr, y_test))


# In[41]:


diff = y_test - rf_preds_tuned
plt.hist(diff, bins=13)


# In[42]:


#feature importance for rf model
importances = rf_tuned.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train_no_corr.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train_no_corr.shape[1]), x_train_no_corr.columns[sorted_indices], rotation=90)
plt.show()


# In[43]:


#XGBoost


# In[44]:


xgb = XGBRegressor()
xgb.fit(x_train_no_corr, y_train)
xgb_preds=xgb.predict(x_test_no_corr)


# In[46]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xgb_preds))
print("MSE:", mean_squared_error(y_test, xgb_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xgb_preds)))
print("Score", xgb.score(x_test_no_corr, y_test))


# In[47]:


xgb_tuned = XGBRegressor(colsample_bytree = 1, gamma = 2, max_depth = 3, min_child_weight = 1, subsample = 1)
xgb_tuned.fit(x_train_no_corr, y_train)
xgb_preds_tuned=xgb_tuned.predict(x_test_no_corr)


# In[48]:


#cross validation of metrics using training data
gkf2=GroupKFold(n_splits=10)
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(xgb_tuned, x_train_no_corr, y_train, scoring = scores, cv=gkf2, groups=group_train)
print(scores)


# In[49]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xgb_preds_tuned))
print("MSE:", mean_squared_error(y_test, xgb_preds_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xgb_preds_tuned)))
print("Score", xgb_tuned.score(x_test_no_corr, y_test))


# In[51]:


#feature importance for xgb tuned model
importances = xgb_tuned.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train_no_corr.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train_no_corr.shape[1]), x_train_no_corr.columns[sorted_indices], rotation=90)
plt.show()


# In[52]:


#svr


# In[84]:


scale = MinMaxScaler()
x_train_scaled = scale.fit_transform(x_train_no_corr)
x_test_scaled = scale.fit_transform(x_test_no_corr)


# In[85]:


svr=SVR()
svr.fit(x_train_scaled, y_train)
svr_preds = svr.predict(x_test_scaled)


# In[111]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_preds))
print("MSE:", mean_squared_error(y_test, svr_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_preds)))
print("Score", svr.score(x_test_scaled, y_test))


# In[113]:


svr_tuned = SVR(kernel = 'poly', gamma = 'scale', degree = 4)
svr_tuned.fit(x_train_scaled, y_train)
svr_pred_tuned = svr_tuned.predict(x_test_scaled)


# In[115]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_pred_tuned))
print("MSE:", mean_squared_error(y_test, svr_pred_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_pred_tuned)))
print("Score", svr_tuned.score(x_test_scaled, y_test))


# In[112]:


from sklearn import model_selection
#cross validation of results using best parameters on training data
gkf2=GroupKFold(n_splits=10)
pipe = make_pipeline(scale, SVR(degree = 4, gamma = 'scale', kernel = 'poly'))
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(pipe, x_train_no_corr, y_train, scoring =scores, cv=gkf2, groups=group_train)
print(scores)


# In[89]:


#trying features selection using random forest for SVR


# In[90]:


from sklearn.feature_selection import SelectFromModel


# In[91]:


sel = SelectFromModel(rf, max_features = 20)
sel.fit(x_train_no_corr, y_train)


# In[92]:


sel.get_support()


# In[93]:


selected_feat= x_train_no_corr.columns[(sel.get_support())]
len(selected_feat)


# In[94]:


print(selected_feat)


# In[95]:


x_train_features = x_train_no_corr[['S1 Mutation', '21A (Delta)', '1Quarter_first_cases',
       '2Quarter_first_cases', 'Omicron', '% Flu Vaccinated']]


# In[96]:


x_test_feature_scale = x_test_no_corr[['S1 Mutation', '21A (Delta)', '1Quarter_first_cases',
       '2Quarter_first_cases', 'Omicron', '% Flu Vaccinated']]


# In[97]:


x_train_feature_scaled = scale.fit_transform(x_train_features)
x_test_feature_scaled = scale.fit_transform(x_test_feature_scale)


# In[98]:


svr_tuned = SVR()
svr_tuned.fit(x_train_feature_scaled, y_train)
svr_preds_tuned = svr_tuned.predict(x_test_feature_scaled)


# In[99]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_preds_tuned))
print("MSE:", mean_squared_error(y_test, svr_preds_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_preds_tuned)))


# In[108]:


svr_tuned = SVR(kernel = 'poly', gamma = 'scale', degree = 2)
svr_tuned.fit(x_train_feature_scaled, y_train)
svr_pred_tuned = svr_tuned.predict(x_test_feature_scaled)


# In[109]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_pred_tuned))
print("MSE:", mean_squared_error(y_test, svr_pred_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_pred_tuned)))


# In[110]:


from sklearn import model_selection
#cross validation of results using best parameters on training data
gkf2=GroupKFold(n_splits=10)
pipe = make_pipeline(scale, SVR(degree = 2, gamma = 'scale', kernel = 'poly'))
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(pipe, x_train_feature_scaled, y_train, scoring =scores, cv=gkf2, groups=group_train)
print(scores)


# In[ ]:




