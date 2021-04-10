#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import chi2_contingency


# https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews#Womens%20Clothing%20E-Commerce%20Reviews.csv

# In[42]:


e_comm_data = pd.read_csv('datasets/E-commerce.csv', index_col=0)

e_comm_data.sample(10)


# In[43]:


e_comm_data.shape


# In[44]:


e_comm_data = e_comm_data[['Recommended IND', 'Rating']]

e_comm_data.head()


# In[45]:


e_comm_data[['Rating']].hist(figsize=(10, 8))


# ### Make a dataframe which having columns `Rating` and contain sum of `frequency` of rating.
# 
# https://towardsdatascience.com/running-chi-square-tests-in-python-with-die-roll-data-b9903817c51b

# In[46]:


df_for_obs = pd.crosstab(e_comm_data['Recommended IND'], e_comm_data['Rating'])

df_for_obs


# ### Returns values `Chi2-test`, `p-value`, `degree of freedom`, and `expected value` 
# 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
# 
# https://en.wikipedia.org/wiki/Contingency_table

# In[47]:


chi2, p_value, degrees_of_freedom, expected_values = chi2_contingency(df_for_obs.values)


# ### `Chi2-test`

# In[48]:


print('Chi2 stats: {}'. format(round(chi2, 3)))


# ### `p-value`

# In[49]:


print('The p-values: {}'.format(p_value))


# ### `Degree of freedom`

# In[50]:


print('The degree of freedom: {}'.format(degrees_of_freedom))


# ### `Expected` occurance of `Rating`

# In[51]:


expected_values


# In[52]:


expected_df = pd.DataFrame({
    '0': expected_values[0],
    '1': expected_values[1],
})


# In[53]:


expected_df


# In[54]:


plt.figure(figsize=(10, 8))

plt.bar(expected_df.index, expected_df['1'], label="Recommended")
plt.bar(expected_df.index, expected_df['0'], label="Not recommended")

plt.legend()


# In[55]:


ratings_recommended = e_comm_data[e_comm_data['Recommended IND'] == 1]

ratings_not_recommended = e_comm_data[e_comm_data['Recommended IND'] == 0]


# In[56]:


ratings_recommended.shape, ratings_not_recommended.shape


# In[62]:


ratings_recommended.sample(10)


# In[63]:


ratings_not_recommended.sample(10)


# In[64]:


ratings_recommended[['Rating']].hist(figsize=(10, 8))


# In[65]:


ratings_not_recommended[['Rating']].hist(figsize=(10, 8))


# In[66]:


plt.figure(figsize=(10, 8))

plt.hist(ratings_recommended['Rating'], label="Recommended")
plt.hist(ratings_not_recommended['Rating'], label="Not recommended")

plt.legend()


# In[ ]:




