#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt


# <b>Dataset: </b> https://www.kaggle.com/altavish/boston-housing-dataset

# * ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# * INDUS - proportion of non-retail business acres per town.
# * CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# * NOX - nitric oxides concentration (parts per 10 million)
# * RM - average number of rooms per dwelling
# * AGE - proportion of owner-occupied units built prior to 1940
# * DIS - weighted distances to five Boston employment centres
# * RAD - index of accessibility to radial highways
# * TAX - full-value property-tax rate per dollar 10,000
# * PTRATIO - pupil-teacher ratio by town
# * LSTAT - lower status of the population-percentage
# * MEDV - Median value of owner-occupied homes in $1000's

# In[2]:


house_data = pd.read_csv('datasets/HousingData.csv')

house_data.head(5)


# In[3]:


house_data.shape


# In[4]:


house_data.columns


# In[5]:


house_data = house_data.drop(['CRIM', 'B'], axis=1)

house_data.head()


# In[6]:


house_data.isnull().sum()


# In[7]:


house_data.dropna(inplace=True, axis=0)


# In[8]:


house_data.shape


# In[9]:


median_price = scipy.median(house_data['MEDV'])

median_price


# In[10]:


house_data['above_median'] = np.where(house_data['MEDV'] > median_price, 1, 0)

house_data.sample(10)


# In[11]:


house_data.to_csv('datasets/house_data_processed.csv', index = False)


# In[12]:


get_ipython().system('ls datasets/')


# ### Bivariate data analysis

# In[13]:


house_data_selected = house_data[['MEDV', 'RM', 'DIS', 'AGE']]

house_data_selected.head(10)


# In[14]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x='AGE', y='MEDV', s=80, 
                data=house_data_selected)

plt.title('House Data')


# In[15]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x='RM', y='MEDV', s=80, 
                data=house_data_selected)

plt.title('House Data')


# In[16]:


sns.pairplot(house_data_selected)

plt.show()


# In[17]:


with sns.axes_style('white'):
    sns.jointplot(x='RM', y='MEDV', data=house_data_selected, kind='hex')
    
    plt.show()


# In[18]:


sns.jointplot(x='AGE', y='MEDV', data=house_data_selected, kind='kde')

plt.show()


# ### Covarience

# In[19]:


house_data_selected_cov = np.cov(house_data_selected.T)

house_data_selected_cov


# ### Correlation

# In[20]:


house_data_selected_corr = np.corrcoef(house_data_selected.T)

house_data_selected_corr


# In[21]:


plt.figure(figsize=(12, 8))

sns.heatmap(house_data_selected_corr, 
            xticklabels=house_data_selected.columns, 
            yticklabels=house_data_selected.columns,
            annot=True)

plt.show()


# ### Linear Regression

# In[22]:


plt.figure(figsize=(12, 8))

sns.lmplot(x='RM', y='MEDV', data=house_data)

plt.title('Salary')


# In[23]:


slope, intercept, r_value, _, _ ,= stats.linregress(house_data['RM'], 
                                                    house_data['MEDV'])


# In[24]:


print('R-square value', r_value**2)


# In[25]:


print('Slope', slope)


# In[26]:


print('Intercept', intercept)


# In[27]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x='RM', y='MEDV', s=100, 
                data=house_data, label='Original')

sns.lineplot(x=house_data['RM'], 
             y=(slope * house_data['RM'] + intercept), 
             color='r', label='Fitted line')

plt.title('Salary')


# In[28]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x='AGE', y='MEDV', s=80, 
                hue='RAD', data=house_data)

plt.title('House Data')


# In[29]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x='RM', y='MEDV', s=80, 
                hue='RAD', data=house_data)

plt.title('House Data')


# In[30]:


X = house_data.drop(['MEDV', 'above_median'], axis=1)

y = house_data['MEDV']


# In[31]:


X.head()


# In[32]:


reg_model = sm.OLS(y, X).fit()

reg_model.params


# In[33]:


reg_model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




