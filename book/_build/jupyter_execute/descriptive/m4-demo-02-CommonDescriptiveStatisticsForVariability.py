#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels

from scipy import stats
from statsmodels.stats import stattools
from statsmodels.stats.weightstats import DescrStatsW


# <b>Datasets : </b> https://www.kaggle.com/kandij/mall-customers

# In[2]:


mall_data = pd.read_csv('datasets/mall_data_processed.csv', index_col=0)

mall_data.head(5)


# In[3]:


mall_data.shape


# In[4]:


mall_data.describe()


# In[5]:


income_descr = DescrStatsW(mall_data['annual_income'])

age_descr = DescrStatsW(mall_data['age'])


# In[6]:


q1_income = income_descr.quantile(0.25)

q3_income = income_descr.quantile(0.75)


# In[7]:


q1_income


# In[8]:


q3_income


# In[9]:


type(q1_income)


# In[10]:


iqr_income = q3_income.loc[0.75] - q1_income.loc[0.25]

iqr_income


# In[11]:


stats.iqr(mall_data['annual_income'])


# In[12]:


stats.iqr(mall_data['annual_income'], interpolation='lower')


# In[13]:


stats.iqr(mall_data['annual_income'], interpolation='higher')


# In[14]:


stats.iqr(mall_data['annual_income'], interpolation='midpoint')


# In[15]:


q1_income_np = np.percentile(mall_data['annual_income'], 25)

q1_income_np


# In[16]:


q3_income_np = np.percentile(mall_data['annual_income'], 75)

q3_income_np


# In[17]:


plt.figure(figsize=(12, 8))

sns.boxplot(mall_data['annual_income'], orient='v')


# In[18]:


plt.figure(figsize=(12, 8))

sns.boxplot(mall_data['spending_score'], orient='v')


# In[19]:


mall_data.head()


# In[20]:


plt.figure(figsize=(12, 8))

sns.boxplot(x='gender', y='annual_income', hue='gender', data=mall_data, orient='v')


# In[21]:


plt.figure(figsize=(12, 8))

sns.boxplot(x='gender', y='spending_score', hue='gender', data=mall_data, orient='v')


# In[22]:


plt.figure(figsize=(12, 8))

sns.boxplot(x='above_average_income', y='spending_score', hue='above_average_income', 
            data=mall_data, orient='v')


# In[23]:


income_descr.var


# In[24]:


age_descr.var


# ### Calculating Standard Deviation

# In[25]:


income_descr.std


# In[26]:


age_descr.std


# #### Describe using stats

# In[27]:


stats.describe(mall_data['annual_income'])


# In[28]:


stats.describe(mall_data['age'])


# In[29]:


listOfSeries = [pd.Series(['Male', 20, 250000, 98, True], index=mall_data.columns ), 
                pd.Series(['Female', 18, 280000, 20, True], index=mall_data.columns ),
                pd.Series(['Male', 78, 20000, 22, True], index=mall_data.columns )
               ]


# In[30]:


mall_updated = mall_data.append(listOfSeries , ignore_index=True)

mall_updated.tail()


# In[31]:


np.ptp(mall_data['annual_income'])


# In[32]:


np.ptp(mall_updated['annual_income'])


# In[33]:


stats.iqr(mall_data['annual_income'], interpolation='midpoint')


# In[34]:


stats.iqr(mall_updated['annual_income'], interpolation='midpoint')


# In[35]:


plt.figure(figsize=(12, 8))

sns.boxplot(mall_updated['spending_score'], orient='v')


# In[36]:


plt.figure(figsize=(12, 8))

sns.boxplot(mall_updated['annual_income'], orient='v')


# In[ ]:




