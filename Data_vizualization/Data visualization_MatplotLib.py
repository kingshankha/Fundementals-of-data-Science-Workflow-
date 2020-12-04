# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:41:12 2020

@author: Shankha
"""
# =============================================================================
# Data Visualization
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("Toyota.csv",index_col=0,na_values=["??","????"])
df.dropna(axis=0,inplace=True)

# =============================================================================
# Scaterplot to see the relationship between two variables
# plt.figure(1)
# =============================================================================
plt.figure(1)
plt.scatter(df["Age"],df["Price"],c='red')
plt.xlabel("Age")
plt.ylabel("Price")
plt.title("cars Age vs cars Price Data")
plt.show()

# =============================================================================
# #### HYSTOGRAM ###########
# =============================================================================

# =============================================================================
# ##graphical representation of different variable data 
# #using bars of different heights
# ##univariable plot 
# ## groups the no into ranges or bins 
# ## the heights depicts the frequency of eatch bin
# ##represents frquency distribtion of numerical variables/datas
# =============================================================================
plt.figure(2)
hist=plt.hist(df["KM"],color="cyan",edgecolor="blue",bins=5)
plt.title("Histogram of kilometer variablem /freq dist")
plt.show()

# =============================================================================
# BAR PLOT
# Bar plot represents the frequency distribution 
# of unique catagories of a catagories of a catagorical variable
# =============================================================================
#%%
plt.figure(3)
FuelType=df["FuelType"].value_counts().values
index=np.arange(len(FuelType))
plt.bar(index,FuelType,color=["blue","green","red"])
plt.xticks(index,df["FuelType"].value_counts().index)
plt.show()




