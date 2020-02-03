#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


sub_1_p = pd.read_csv('./output/submission_1020.csv')
sub_2_p = pd.read_csv('./output/submission_1021.csv')
sub_3_p = pd.read_csv('./output/submission_12345.csv')
sub_4_p = pd.read_csv('./output/submission_1234.csv')
sub_5_p = pd.read_csv('./output/submission_2017.csv')
sub_6_p = pd.read_csv('./output/submission_4242.csv')
sub_7_p = pd.read_csv('./output/submission_77777.csv')
sub_8_p = pd.read_csv('./output/submission_8895.csv')


# In[4]:


total_type_p = pd.DataFrame()
total_type_p['sub_1020'] = sub_1_p['scalar_coupling_constant']
total_type_p['sub_1021'] = sub_2_p['scalar_coupling_constant']
total_type_p['sub_12345'] = sub_3_p['scalar_coupling_constant']
total_type_p['sub_1234'] = sub_4_p['scalar_coupling_constant']
total_type_p['sub_2017'] = sub_5_p['scalar_coupling_constant']
total_type_p['sub_4242'] = sub_6_p['scalar_coupling_constant']
total_type_p['sub_77777'] = sub_7_p['scalar_coupling_constant']
total_type_p['sub_8895'] = sub_8_p['scalar_coupling_constant']


# In[5]:


total_type_p['type'] = pd.read_csv('type.csv')['type']


# In[7]:


types = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']


# In[8]:


cv = pd.read_csv('./cv.csv')


# In[9]:


cv = cv.iloc[[13, 13, 1, 0, 7, 8, 6, 11]][types].reset_index(drop=True)


# In[10]:


cv = cv.astype(float)


# In[11]:


import numpy as np


# In[12]:


def new_softmax(a) : 
    #c = np.max(a) # 최댓값
    #a = a / (-a).rank()
    exp_a = np.exp(a) #
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# In[13]:


cv_softmax = (-cv).apply(new_softmax, axis=0).T


# In[15]:


frame_list = []
for typ in types:
    frame_list.append((total_type_p[total_type_p.type == typ].drop(['type'], axis=1) * cv_softmax.loc[typ].values).sum(axis=1).to_frame())


# In[16]:


total_type_p['new'] = pd.concat(frame_list).sort_index().values# / 3


# In[17]:


sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission.scalar_coupling_constant = total_type_p.new
sample_submission.to_csv('submission_type_softmax_pseudo_final.csv', index=False)


