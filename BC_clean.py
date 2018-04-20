# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:02:01 2017

@author: admin
"""
#This code reads breast cancer data, searches for outliers, normalizes the data
#and outputs a cleaned version of same data. 

#load necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

#read the Data and change text variables to numerical 
df = pd.read_csv("BreastCancerData.csv",header = 0)
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

#delete variables that are un-necessary or show NAN values
del df['id']
del df['Unnamed: 32']

#clean exmaples having values away from 3rd standard deviation from mean 
df_clean = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

#output cleaned version of the data          
#==============================================================================
df_clean.to_csv('BreastCancerData_Clean.csv', header= True, index = False)
#==============================================================================

#normalize the data for vislualisations
df_norm = (df_clean - df_clean.min()) / (df_clean.max() - df_clean.min()) 

#plot the violt graph to show data distribution against responce variable
mbc = pd.melt(df_norm, "diagnosis", var_name="Variables")
fig, ax = plt.subplots(figsize=(10,5))
p = sns.violinplot(ax = ax, x="Variables", y="value", hue="diagnosis", split = True, data=mbc, inner = 'quartile', palette = 'Set1_r');
p.set_xticklabels(rotation = 90, labels = list(df_norm.columns));