#!/usr/bin/env python
# coding: utf-8

# ## Two-way ANOVA
# https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/anova/how-to/two-way-anova/interpret-the-results/key-results/#step-1-determine-whether-the-main-effects-and-interaction-effect-are-statistically-significant

# In[7]:


import pandas as pd
import numpy as numpy

import matplotlib.pyplot as plt

from scipy import stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[9]:


bike_sharing_data = pd.read_csv('datasets/bike_sharing_data_processed.csv')

bike_sharing_data.head()


# In[12]:


bike_sharing_data['weathersit'].unique()


# In[13]:


bike_sharing_data['season'].unique()


# ### Mean of `windspeed` group by `weathersit`

# In[35]:


rp.summary_cont(bike_sharing_data.groupby(['weathersit']))['cnt']


# In[36]:


bike_sharing_data.boxplot(column=['cnt'], by='weathersit', figsize=(12, 8))


# In[37]:


rp.summary_cont(bike_sharing_data.groupby(['season']))['cnt']


# In[38]:


bike_sharing_data.boxplot(column=['cnt'], by='season', figsize=(12, 8))


# ### Find F-statistics by ols model
# https://www.statsmodels.org/stable/examples/notebooks/generated/interactions_anova.html#Two-way-ANOVA

# In[39]:


model = ols('cnt ~ C(weathersit)', bike_sharing_data).fit()

print(model.summary())


# In[40]:


model = ols('cnt ~ C(season)', bike_sharing_data).fit()

print(model.summary())


# In[41]:


model = ols('cnt ~ C(weathersit) + C(season)', bike_sharing_data).fit()

print(model.summary())


# In[42]:


sm.stats.anova_lm(model)


# In[43]:


model = ols('cnt ~ C(weathersit) * C(season)', bike_sharing_data).fit()

print(model.summary())


# In[44]:


sm.stats.anova_lm(model)


# In[ ]:





# In[ ]:




