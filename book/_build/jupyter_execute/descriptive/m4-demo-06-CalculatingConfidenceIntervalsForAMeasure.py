#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


housing_data = pd.read_csv('datasets/house_data_processed.csv')

housing_data.head()


# In[3]:


housing_data.shape


# In[4]:


plt.figure(figsize=(12, 8))

sns.boxplot(x='MEDV', data=housing_data, orient='v')

plt.show()


# In[5]:


plt.figure(figsize=(12, 8))

sns.distplot(housing_data['MEDV'], rug=True, hist=False)

plt.show()


# In[6]:


mean = np.mean(housing_data['MEDV'])
std = np.std(housing_data['MEDV'])


# In[7]:


mean, std


# #### For 68% confidence Interval

# In[8]:


conf_int_low, conf_int_high = scipy.stats.norm.interval(0.68, loc=mean, scale=std)

conf_int_low, conf_int_high


# In[9]:


plt.figure(figsize=(12, 8))

sns.distplot(housing_data['MEDV'], rug=True, kde=True, hist=False)

plt.stem([mean], 
         [0.06], linefmt = 'C1', 
         markerfmt = 'C1', label = 'mean', use_line_collection=True)

plt.stem([conf_int_low], 
         [0.06], linefmt = 'C2', 
         markerfmt = 'C2', label = '95% CI High', use_line_collection=True)

plt.stem([conf_int_high], 
         [0.06], linefmt = 'C3', 
         markerfmt = 'C3', label = '95% CI Low', use_line_collection=True)

plt.xlabel('MEDV')
plt.legend()
plt.show()


# ### For 90% confidence interval

# In[10]:


conf_int_low, conf_int_high = scipy.stats.norm.interval(0.90, loc=mean, scale=std)

conf_int_low, conf_int_high


# In[11]:


plt.figure(figsize=(12, 8))

sns.distplot(housing_data['MEDV'], rug=True, kde=True, hist=False)

plt.stem([mean], 
         [0.06], linefmt = 'C1', 
         markerfmt = 'C1', label = 'mean', use_line_collection=True)

plt.stem([conf_int_low], 
         [0.06], linefmt = 'C2', 
         markerfmt = 'C2', label = '95% CI High', use_line_collection=True)

plt.stem([conf_int_high], 
         [0.06], linefmt = 'C3', 
         markerfmt = 'C3', label = '95% CI Low', use_line_collection=True)

plt.xlabel('MEDV')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




