#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[13]:


dataclean = pd.read_csv("df_var_data.csv")
dataclean = pd.DataFrame(dataclean)
dataclean


# In[16]:


ap = pd.read_csv('airport.csv')
ap = pd.DataFrame(ap)
ap


# In[18]:


dataclean['State'] = dataclean['State'].astype(str)
ap['State'] = ap['State'].astype(str)


# In[19]:


merge = dataclean.merge(ap,how="left")


# In[20]:


merge.to_csv('merge.csv')

