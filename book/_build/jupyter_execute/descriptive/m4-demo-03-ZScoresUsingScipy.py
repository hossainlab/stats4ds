#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import pandas as pd
import seaborn as sns

from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt


# In[2]:


salary_data = pd.read_csv('datasets/Salary_Data.csv')

salary_data.head()


# In[3]:


num_records = salary_data.shape[0]

num_records


# In[4]:


import math 

def z_score(series, value):
    mean = float(series.mean())
    
    variance = float(series.var())
    std = float(math.sqrt(variance))
    
    return float((value - mean) / std)


# In[5]:


salary_data['z_score_salary'] = [1.00] * num_records

salary_data.head()


# In[6]:


for i in range(len(salary_data['Salary'])):
    salary_data['z_score_salary'][i] = z_score(salary_data['Salary'], 
                                               salary_data['Salary'][i])


# In[7]:


salary_data['Salary'].mean()


# In[8]:


salary_data.sample(10)


# In[9]:


salary_data.describe()


# In[10]:


salary_data['z_score_yrexp'] = [1.00] * num_records

salary_data.head()


# In[11]:


salary_data['z_score_yrexp'] = scipy.stats.zscore(salary_data['YearsExperience'])

salary_data.sample(10)


# In[12]:


salary_data.describe()


# In[13]:


plt.figure(figsize=(12, 8))

sns.kdeplot(salary_data['z_score_yrexp'])

plt.title('z score yrexp')
plt.ylabel('density')


# In[14]:


plt.figure(figsize=(12, 8))

sns.kdeplot(salary_data['z_score_salary'])

plt.title('z score salary')
plt.ylabel('density')


# In[15]:


plt.figure(figsize=(12, 8))

sns.distplot(salary_data['YearsExperience'], bins=10)

plt.title('Year Experience')
plt.ylabel('density')


# In[16]:


plt.figure(figsize=(12, 8))

sns.distplot(salary_data['Salary'], bins=10)

plt.title('Salary')
plt.ylabel('density')


# In[17]:


mu_yrexp =  scipy.mean(salary_data['YearsExperience'])

sigma_yrexp = scipy.std(salary_data['YearsExperience'])


# In[18]:


fig, ax = plt.subplots(figsize=(12, 6))

num_bins = 15

n, bins, patches = ax.hist(salary_data['YearsExperience'], 
                           num_bins, density = 1)

y = norm.pdf(bins, mu_yrexp, sigma_yrexp)

ax.plot(bins, y, 'o-')

fig.tight_layout()
plt.show()


# In[19]:


mu_salary =  scipy.mean(salary_data['Salary'])

sigma_salary = scipy.std(salary_data['Salary'])


# In[20]:


num_bins = 30

fig, ax = plt.subplots(figsize=(12, 6))

n, bins, patches = ax.hist(salary_data['Salary'], 
                           num_bins, density = 1)

y = norm.pdf(bins, mu_salary, sigma_salary) 
ax.plot(bins, y, 'o-')

fig.tight_layout()
plt.show()


# In[21]:


print("Skew of Experience : ", stats.skew(salary_data['YearsExperience']))

print("Kurtosis of Experience : ", stats.kurtosis(salary_data['YearsExperience']))


# In[22]:


print("Skew of z_score of Experience : ", stats.skew(salary_data['z_score_yrexp']))

print("Kurtosis of z_score of Experience : ", stats.kurtosis(salary_data['z_score_yrexp']))


# In[23]:


print("Skew of salary : ", stats.skew(salary_data['Salary']))

print("Kurtosis of salary : ", stats.kurtosis(salary_data['Salary']))


# In[24]:


print("Skew of z_score of Salary : ", stats.skew(salary_data['z_score_salary']))

print("Kurtosis of z_score of Salary : ", stats.kurtosis(salary_data['z_score_salary']))


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




