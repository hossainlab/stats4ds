#!/usr/bin/env python
# coding: utf-8

# # Reading and Exploring Data using Pandas 

# In[1]:


import pandas as pd 


# In[3]:


# read data 
df = pd.read_csv('../data/diabetes.csv')


# In[4]:


# examine first few  rows 
df.head()


# In[5]:


# examine last few rows 
df.tail() 


# In[6]:


# shape 
df.shape


# In[7]:


# columns 
df.columns


# In[9]:


# data types 
df.dtypes


# In[8]:


# info
df.info() 

