#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[1]:


import pickle
import pandas as pd


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[3]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[6]:


filename = 'fhv_tripdata_2021-02.parquet'

df = read_data(filename)


# In[7]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[13]:


## Question 1


# In[14]:


from statistics import mean

mean(y_pred)


# In[15]:


## Question 2


# In[12]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[ ]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:


## Question 3


# jupyter nbconvert starter.ipynb --to script

# In[16]:


## Question 4


# In[17]:


## Question 5


# In[ ]:


## Question 6

