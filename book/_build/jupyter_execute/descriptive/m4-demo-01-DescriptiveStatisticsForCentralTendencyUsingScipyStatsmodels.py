#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U scipy')


# In[2]:


get_ipython().system('pip install -U statsmodels')


# In[3]:


import scipy 
import statsmodels


# In[4]:


scipy.__version__


# In[5]:


statsmodels.__version__


# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW


# In[7]:


salary_data = pd.read_csv('datasets/Salary_Data.csv')

salary_data.sample(10)


# In[8]:


salary_data.shape


# In[9]:


salary_data.isnull().sum()


# In[10]:


min_exp = np.min(salary_data['YearsExperience'])

min_exp


# In[11]:


max_exp = np.max(salary_data['YearsExperience'])

max_exp


# In[12]:


min_salary = np.min(salary_data['Salary'])

min_salary


# In[13]:


max_salary = np.max(salary_data['Salary'])

max_salary


# In[14]:


range_of_exp = np.ptp(salary_data['YearsExperience'])

range_of_exp


# In[15]:


range_of_salary = np.ptp(salary_data['Salary'])

range_of_salary


# In[16]:


salary = salary_data['Salary']

salary.head(10)


# In[17]:


sorted_salary = salary.sort_values().reset_index(drop=True)

sorted_salary.head(10)


# In[18]:


salary_mean = scipy.mean(salary_data['Salary'])

salary_mean


# In[19]:


exp_stats = DescrStatsW(salary_data['YearsExperience'])

exp_stats.mean


# In[20]:


salary_median = scipy.median(sorted_salary)

salary_median


# In[21]:


salary_median = scipy.median(salary_data['Salary'])

salary_median


# In[22]:


exp_stats.quantile(0.5)


# In[23]:


plt.figure(figsize=(12, 8))

sns.distplot(salary_data['Salary'])


# In[24]:


plt.figure(figsize=(12, 8))

sns.distplot(salary_data['YearsExperience'])


# In[25]:


plt.figure(figsize=(12, 8))

sns.distplot(salary_data['Salary'])

plt.axvline(salary_mean, color='r', label='mean')
plt.axvline(salary_median, color='b', label='median')

plt.legend()


# In[26]:


plt.figure(figsize=(12, 8))

sns.barplot(x='YearsExperience', y='Salary', data=salary_data)

plt.axhline(salary_mean, color='r', label='mean')
plt.axhline(salary_median, color='b', label='median')

plt.legend()

plt.show()


# In[27]:


listOfSeries = [pd.Series([20, 250000], index=salary_data.columns ), 
                pd.Series([25, 270000], index=salary_data.columns ), 
                pd.Series([30, 320000], index=salary_data.columns )]


# In[28]:


salary_updated = salary_data.append(listOfSeries , ignore_index=True)

salary_updated.tail()


# In[29]:


salary_updated_mean = scipy.mean(salary_updated['Salary'])

salary_updated_mean


# In[30]:


salary_mean


# In[31]:


salary_updated_median = scipy.median(salary_updated['Salary'])

salary_updated_median


# In[32]:


salary_median


# In[33]:


plt.figure(figsize=(12, 8))

sns.distplot(salary_updated['Salary'])

plt.axvline(salary_updated_mean, color='r', label='mean')
plt.axvline(salary_updated_median, color='b', label='median')

plt.legend()


# In[34]:


plt.figure(figsize=(12, 8))

sns.distplot(salary_data['Salary'], hist_kws={'alpha':0.2}, color='grey')
sns.distplot(salary_updated['Salary'], hist_kws={'alpha':0.8}, color='green')

plt.axvline(salary_mean, color='grey', label='mean')
plt.axvline(salary_updated_mean, color='green', label='median')

plt.legend()


# In[35]:


plt.figure(figsize=(12, 8))

sns.distplot(salary_data['Salary'], hist_kws={'alpha':0.2}, color='grey')
sns.distplot(salary_updated['Salary'], hist_kws={'alpha':0.8}, color='green')

plt.axvline(salary_median, color='grey', label='mean')
plt.axvline(salary_updated_median, color='green', label='median')

plt.legend()


# In[36]:


plt.figure(figsize=(12, 8))

sns.barplot(x='YearsExperience', y='Salary', data=salary_updated)

plt.axhline(salary_updated_mean, color='r', label='mean')
plt.axhline(salary_updated_median, color='b', label='median')
plt.xticks(rotation=90)

plt.legend()

plt.show()


# In[37]:


stats.mode(salary_data['YearsExperience'])


# In[38]:


stats.mode(salary_data['Salary'])


# In[39]:


plt.figure(figsize=(12, 8))

sns.countplot(salary_data['YearsExperience'])


# In[40]:


plt.figure(figsize=(12, 8))

sns.countplot(salary_data['Salary'])

plt.xticks(rotation=90)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




