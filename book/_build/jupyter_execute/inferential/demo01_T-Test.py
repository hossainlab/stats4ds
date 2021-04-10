#!/usr/bin/env python
# coding: utf-8

# In[53]:


get_ipython().system('pip install scipy')


# In[54]:


get_ipython().system('pip install researchpy')


# In[1]:


import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

import researchpy as rp
from scipy import stats


# https://www.kaggle.com/lakshmi25npathi/bike-sharing-dataset

# In[2]:


bike_sharing_data = pd.read_csv('datasets/day.csv')

bike_sharing_data.shape


# In[3]:


bike_sharing_data.head()


# In[4]:


bike_sharing_data = bike_sharing_data[['season', 
                                       'mnth', 
                                       'holiday', 
                                       'workingday', 
                                       'weathersit', 
                                       'temp',
                                       'cnt']]


# In[5]:


bike_sharing_data.to_csv('datasets/bike_sharing_data_processed.csv', index=False)


# In[6]:


bike_sharing_data.head()


# In[7]:


bike_sharing_data['season'].unique()


# In[8]:


bike_sharing_data['workingday'].unique()


# In[9]:


bike_sharing_data['holiday'].unique()


# In[10]:


bike_sharing_data['weathersit'].unique()


# In[11]:


bike_sharing_data['temp'].describe()


# In[12]:


bike_sharing_data.shape


# In[13]:


bike_sharing_data.groupby('workingday')['cnt'].describe()


# In[14]:


bike_sharing_data.boxplot(column=['cnt'], by='workingday', figsize=(12, 8))


# In[15]:


sample_01 = bike_sharing_data[(bike_sharing_data['workingday'] == 1)]

sample_02 = bike_sharing_data[(bike_sharing_data['workingday'] == 0)]


# In[16]:


sample_01.shape, sample_02.shape


# In[17]:


sample_01 = sample_01.sample(231)

sample_01.shape, sample_02.shape


# ## The hypothesis being tested
# 
# * __Null hypothesis (H0): u1 = u2, which translates to the mean of `sample_01` is equal to the mean of `sample 02`__
# * __Alternative hypothesis (H1): u1 ? u2, which translates to the means of `sample01` is not equal to `sample 02`__

# ### Homogeneity of variance
# Of these tests, the most common assessment for homogeneity of variance is Levene's test. The Levene's test uses an F-test to test the null hypothesis that the variance is equal across groups. A p value less than .05 indicates a violation of the assumption.
# 
# https://en.wikipedia.org/wiki/Levene%27s_test
# 
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.levene.html
# 
# To know, [Click here](https://en.wikipedia.org/wiki/Levene%27s_test) why we test for levene's test?

# In[18]:


stats.levene(sample_01['cnt'], sample_02['cnt'])


# ## Normal distribution  of residuals
# 
# ### Checking difference between two pair points
# 
# https://pythonfordatascience.org/independent-t-test-python/

# In[19]:


diff = scale(np.array(sample_01['cnt']) - np.array(sample_02['cnt'], dtype=np.float))

plt.hist(diff)


# ### Checking for normality by Q-Q plot graph
# 
# https://www.statisticshowto.datasciencecentral.com/assumption-of-normality-test/

# In[20]:


plt.figure(figsize=(12, 8))

stats.probplot(diff, plot=plt, dist='norm')

plt.show()


# ### Checking normal distribution by `shapiro method`
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
# 
# https://stats.stackexchange.com/questions/15696/interpretation-of-shapiro-wilk-test

# In[21]:


stats.shapiro(diff)


# __Note:-__[See here](https://stats.stackexchange.com/questions/15696/interpretation-of-shapiro-wilk-test)
# 
# W test statistic and the second value is the p-value. Since the test statistic does not produce a significant p-value, the data is indicated to be normally distributed
# 
# The data met all the assumptions for the t-test which indicates the results can be trusted and the t-test is an appropriate test to be used.

# ### Independent t-test by using `scipy.stats`

# In[22]:


stats.ttest_ind(sample_01['cnt'], sample_02['cnt'])


# ### Independent t-test using `researchpy`
# 
# https://researchpy.readthedocs.io/en/latest/ttest_documentation.html

# In[23]:


descriptives, results = rp.ttest(sample_01['cnt'], sample_02['cnt'])


# In[24]:


descriptives


# In[25]:


print(results)


# In[26]:


bike_sharing_data.head()


# In[27]:


bike_sharing_data[['temp']].boxplot(figsize=(12, 8))


# In[28]:


bike_sharing_data['temp_category'] =     bike_sharing_data['temp'] > bike_sharing_data['temp'].mean()


# In[29]:


bike_sharing_data.sample(10)


# In[30]:


bike_sharing_data.groupby('temp_category')['cnt'].describe()


# In[31]:


bike_sharing_data.boxplot(column=['cnt'], by='temp_category', figsize=(12, 8))


# In[32]:


sample_01 = bike_sharing_data[(bike_sharing_data['temp_category'] == True)]

sample_02 = bike_sharing_data[(bike_sharing_data['temp_category'] == False)]


# In[33]:


sample_01.shape, sample_02.shape


# In[34]:


sample_01 = sample_01.sample(364)

sample_01.shape, sample_02.shape


# In[35]:


stats.levene(sample_01['cnt'], sample_02['cnt'])


# In[36]:


diff = scale(np.array(sample_01['cnt']) - np.array(sample_02['cnt']))
plt.hist(diff)


# In[37]:


plt.figure(figsize=(12, 8))
stats.probplot(diff, plot=plt)
plt.show()


# In[38]:


stats.shapiro(diff)


# In[39]:


stats.ttest_ind(sample_01['cnt'], sample_02['cnt'])


# In[43]:


descriptives, results = rp.ttest(sample_01['cnt'], sample_02['cnt'], equal_variances=False)


# In[44]:


descriptives


# In[45]:


print(results)


# In[ ]:




