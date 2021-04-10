#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

from scipy import stats
import researchpy as rp


# https://github.com/Opensourcefordatascience/Data-sets/blob/master/blood_pressure.csv
# 
# In this dataset fictitious and contains blood pressure readings before and after an intervention. These are variables “bp_before” and “bp_after”.

# In[3]:


bp_reading = pd.read_csv('datasets/blood_pressure.csv')


# In[4]:


bp_reading.sample(10)


# In[5]:


bp_reading.shape


# In[6]:


bp_reading.describe().T


# In[7]:


bp_reading[['bp_before', 'bp_after']].boxplot(figsize=(12, 8))


# ## The hypothesis being tested

# * __Null hypothesis (H0): u1 = u2, which translates to the mean of sample 01 is equal to the mean of sample 02__
# * __Alternative hypothesis (H1): u1 ? u2, which translates to the means of sample 01 is not equal to sample 02__ 

# ## Assumption check 
# 
# * The samples are independently and randomly drawn
# * The distribution of the residuals between the two groups should follow the normal distribution
# * The variances between the two groups are equal

# In[8]:


stats.levene(bp_reading['bp_after'], bp_reading['bp_before'])


# In[9]:


bp_reading['bp_diff'] = scale(bp_reading['bp_after'] - bp_reading['bp_before'])


# In[10]:


bp_reading[['bp_diff']].head()


# In[11]:


bp_reading[['bp_diff']].hist(figsize=(12, 8))


# ### Checking Normal distribution by Q-Q plot graph
# https://www.statisticshowto.datasciencecentral.com/assumption-of-normality-test/

# In[12]:


plt.figure(figsize=(15, 8))
stats.probplot(bp_reading['bp_diff'], plot=plt)

plt.title('Blood pressure difference Q-Q plot')
plt.show()


# **Note:-** The corresponding points are lies very close to line that means are our sample data sets are normally distributed

# ### Checking Normal distribution by method of `Shapiro stats`
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html

# In[13]:


stats.shapiro(bp_reading['bp_diff'])


# In[14]:


stats.ttest_rel(bp_reading['bp_after'], bp_reading['bp_before'])


# **Note:-** __Here, `t-test = -3.337` and `p-value = 0.0011` since p-value is less than the significant value hence null-hypothesis is rejected`(Alpha = 0.05)`__

# ### T-test using `researchpy`
# https://researchpy.readthedocs.io/en/latest/ttest_documentation.html

# In[23]:


rp.ttest(bp_reading['bp_after'], bp_reading['bp_before'], 
         paired = True, equal_variances=False)


# In[ ]:




