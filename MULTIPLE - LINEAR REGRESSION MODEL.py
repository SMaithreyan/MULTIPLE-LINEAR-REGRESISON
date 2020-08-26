#!/usr/bin/env python
# coding: utf-8

# In[23]:


## house price prediction based on several factor , area , age of house ,no. of bedrooms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets , linear_model

datas = pd.read_csv('C:\\Users\\raghu\\Downloads\\house_prices.csv')

datas.head()


# In[24]:


datas.bedrooms.median()


# In[27]:


datas.bedrooms = datas.bedrooms.fillna(datas.bedrooms.median())
datas


# In[28]:


reg = linear_model.LinearRegression()
reg.fit(datas.drop('price',axis='columns'),datas.price)


# In[30]:


reg.intercept_


# In[31]:


reg.coef_


# In[32]:


# predict a house with area 3000 , bedrooms 3 , age 40 

reg.predict([[3000, 3, 40]])


# In[34]:


# predict a house with area 2500 , bedrooms 4 , age 5 

reg.predict([[2500, 4, 5]])


# In[ ]:




