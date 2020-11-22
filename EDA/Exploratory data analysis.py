# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:44:32 2020

@author: Shankha
"""
# =============================================================================
# ##  Exploratory data analyis 
# 
# =============================================================================
# =============================================================================
# ## importing Dataset
# 
# =============================================================================
import os
import pandas as pd

df=pd.read_csv("Toyota.csv",index_col=0,na_values=["??","????"])
df2=df.copy()

# =============================================================================
# ## Frequency Table  Uni variable analysis
# 
# =============================================================================
freq_table=pd.crosstab(index=df2["FuelType"],columns='count',dropna=True)

# =============================================================================
# ##  Two way table 
# 
# ##relationship between twe catagorical variable;
# 
# =============================================================================

Two_way_table=pd.crosstab(index=df2["Automatic"],columns
            =df2["FuelType"],dropna=True)

# =============================================================================
# ##Joint Probability
# ##probablity  of two indipendent event happening at the same time
# =============================================================================

Joint_probability=pd.crosstab(index=df2["Automatic"],columns=df2["FuelType"],dropna=True,normalize=True)

# =============================================================================
# ## MArginal Probability
# ##Marginal probability is the probability of an event irrespective of the outcome of another variable.
# =============================================================================

marginal_probability=pd.crosstab(index=df2["Automatic"],columns=df["FuelType"],margins=True,dropna=True,normalize=True)

# ============== ===============================================================
# ## conditional probability
# ## probaility of occurance of one event given another event has already occured 
# ## normalize="index" to get the rowsum=1;
# ##1 given the type of gearbox
# =============================================================================
conditional_probability=pd.crosstab(index=df2["Automatic"],columns=df2["FuelType"],dropna=True,normalize="index",margins=True)

# =============================================================================
# Normalization turns quantitetive values to proportions 
# =============================================================================

# =============================================================================
# ##2 given the type of fuel 
# =============================================================================


conditional_probability2=pd.crosstab(index=df2["Automatic"],columns=df2["FuelType"],dropna=True,normalize="columns",margins=True)


# =============================================================================
# ####  COrrelation #####
# 
# ##Sterngth of association between two variables 
# 
# ## correlation constants(pearson, pvalues) and correlation visualization
# 
# ## pearson only works with quantitetive variables
# 
# ###  NOTE  ##
# 
# ## extract numerical variables from the pandas dataframe 
# 
# =============================================================================
numerical_data_vars=df2.select_dtypes(exclude=[object])

corr_matrix=numerical_data_vars.corr(method="pearson")