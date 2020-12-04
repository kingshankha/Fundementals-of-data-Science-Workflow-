# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:22:16 2020

@author: Shankha Mullick
"""
# =============================================================================
#$$$$$$$$$$$$$ Seaborn $$$$$$$$$$$$$$$$$$$$$$ 
# 
# built on matplotlib
# high level interface for high level statistical plots
# 
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Toyota.csv",index_col=0,na_values=["??","????"])
df.dropna(axis=0,inplace=True)

# =============================================================================
# #Scatterplot
# =============================================================================

# =============================================================================
# regplot is use to visualize the relationsip between two variables (indipendent vs dependent or independent)
# the reg plot or regression plot can be used to visualize the trend betwwn variable pair
#it uses the linear regression to plot the data 
# =============================================================================
plt.figure(1)
sns.set(style="darkgrid")
sns.regplot(df["Age"],df["Price"])
plt.xlabel("Age")
plt.ylabel("Price")
plt.title(" Age vs Price Regression fit")
# =============================================================================
# ### to exclude regresion fit line and use it as a scatterplot 
# =============================================================================
plt.figure(2)
sns.regplot(df["Age"],df["Price"],fit_reg=False)
plt.title(" Age vs Price scatter")

## to chage the marker type to star or other
plt.figure(3)
sns.regplot(df["Age"],df["Price"],fit_reg=False,marker="*")
plt.legend("marker", loc="best")
# =============================================================================
# Scatterplot by including catagorical variable
# =============================================================================
# =============================================================================
# using hue parameter in sns.lmplot we can catagorize or classify \
# the data points using some sets of colors to represent the catagory
# =============================================================================

# =============================================================================
# Scatterplot of price Vs age by FuelType
# =============================================================================
plt.figure(4)
sns.lmplot(x="Age",y="Price",data=df,fit_reg=False,hue="FuelType",palette="Set1")
plt.title("Scatterplot of price Vs age by FuelType")
# =============================================================================
# Histogram
# to visualize the frequency distributio of any continuous variable
# =============================================================================
plt.figure(6)
sns.distplot(df["Age"]) 
plt.show()

## the fit line shows the kernel density estimation line
## it is a non parametric way to determine probability density function 

### without kernel density estimation and specifing the bin nos 
plt.figure(7)
plt.title("frequency distribution without kernel density estimation")
sns.distplot(df["Age"],kde=False,bins=5)


# =============================================================================
# ##BAR PLOT
# to visualize frequency distribution  ofcatagorical variables
# =============================================================================
plt.figure(8)
plt.title("Bar plot")
sns.countplot(x="FuelType",data=df)

### Group barplot

plt.figure(9)
plt.title("Group bar plot")
sns.countplot(x="FuelType",data=df, hue="Automatic")
# =============================================================================
# BOX AND WHISKERS PLOT
# =============================================================================
# The five number summary includes 5 items: The minimum.
# Q1 (the first quartile, or the 25% mark).
# The median.
# Q3 (the third quartile, or the 75% mark).
# The maximum.
# =============================================================================
# =============================================================================
# =============================================================================
#Boxplot to visualize the summery of the veriable min,max,median,interquartile ranges
# can be used to spot outliers 
# =============================================================================
plt.figure(10)
plt.title("Box and Whiskers plot")
sns.boxplot(y=df["Price"])
## y or x is specified to plot on y or x axis
# =============================================================================
# ####&&&& BOX AND WHISKERS PLOT TO VISUALIZE THE RELATIONSHIP BETWEEN 
#  CATAGORICAL AND NUMERICAL VARIABLE
# =============================================================================
plt.figure(11)
plt.title("Box and Whiskers plot catagorical vs numerical")
sns.boxplot(x=df["FuelType"],y=df["Price"])


## GROUP box and wiskers plot 
## visualization of relationship between two variables given another variable
## or catagorized by another variable value

plt.figure(12)
plt.title("Group Box and Wiskers Plot")
sns.boxplot(x="FuelType",y="Price",data=df,hue="Automatic")


##MULTIPLE PLOT IN A SINGLE window USING SUBPLOTS
## Splitting the ploting window in various sections 
## to visualize two or more plottings  in single window
plt.figure(13)
f,(ax_box,ax_hist)=plt.subplots(2,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(df["Price"],ax=ax_box)
sns.distplot(df["Price"],ax=ax_hist,kde=False)

##PAIRWISE PLOTS
##USED TO pairwise relationship between all variables in the dataset
## Creates scatterplot for joint relationship and histogram for univarient distribution 
## In a single plot window
## also hue can be use tom catagorise the plots gien a variabe
plt.figure(14)
sns.pairplot(data=df,kind="scatter",hue="FuelType")



