#!/usr/bin/env python
# coding: utf-8

# ## One-way ANOVA
# 
# https://www.kaggle.com/lakshmi25npathi/bike-sharing-dataset

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
import researchpy as rp

from statsmodels.formula.api import ols 


# In[2]:


bike_sharing_data = pd.read_csv('datasets/bike_sharing_data_processed.csv')

bike_sharing_data.head()


# In[17]:


bike_sharing_data.shape


# In[18]:


bike_sharing_data['weathersit'].unique()


# In[19]:


bike_sharing_data.groupby('weathersit')['cnt'].describe().T


# In[20]:


bike_sharing_data.boxplot(column=['cnt'], by='weathersit', figsize=(12, 8))


# ## The hypothesis being tested
# https://statisticsbyjim.com/anova/post-hoc-tests-anova/

# * __H0: No difference between means, i.e. ?x1 = ?x2 = ?x3__
# * __Ha: Difference between means exist somewhere, i.e. ?x1 ? ?x2 ? ?x3, or ?x1 = ?x2 ? ?x3, or ?x1 ? ?x2 = ?x3__

# ### ANOVA with `scipy.stats`

# In[21]:


stats.f_oneway(bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 1],
               bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 2],
               bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 3],)


# ### ANOVA with `statsmodels`
# https://www.statsmodels.org/stable/examples/notebooks/generated/interactions_anova.html

# In[22]:


result = ols('cnt ~ C(weathersit)', data = bike_sharing_data).fit()


# In[23]:


print(result.summary())


# ### Post - hoc test by using `Tukey's method`
# https://www.statisticshowto.datasciencecentral.com/tukey-test-honest-significant-difference/

# In[24]:


from statsmodels.stats.multicomp import MultiComparison

mul_com = MultiComparison(bike_sharing_data['cnt'], bike_sharing_data['weathersit'])

mul_result = mul_com.tukeyhsd()

print(mul_result)


# In[ ]:




