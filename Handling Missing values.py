# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:19:32 2020

@author: Shankha
"""
# =============================================================================
# Handling Missing values
# =============================================================================

import  pandas as pd

dF=pd.read_csv("Toyota.csv",index_col=0,na_values=["??","????"])
df=dF.copy()
df_pipeline=df.copy()
# =============================================================================
# ##na_values=[] replaces special char or missing values with nan values 
# =======================================================================

# =============================================================================
# checking for nan values in the data frame using isnull or isna()
# =============================================================================

print(df.isna().sum()) 
# using sum to get the count of nan for eatch variable

# now we will get the variables which has attleast 1 misssing value as a pandas series,

missing_df=df[df.isnull().any(axis=1)]

###        +++Conclusion++++
##  if we want to drop all the rows which has missing vales 
## then we are lossing lots of valueabe data 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# =============================================================================
# aproach to replace or fill in missing values 
# 1 drop row 
# 2 drop whole variable
# 3 impute the data using mean/median value 
# 4 impute the data using modevale for catagorical data 
# =============================================================================

### Now we will use descriptie statistics to judge wether  the data
### should be imputed with mean or median value in case of numerical variable

## describe method summerises the central tendency , dispersion and shape of dataset 
## distribution ,excluding nan values \
    
    
# =============================================================================
# if we have a extreme value in the variable so imputing with mean values can be misleading 
# beacuse mean values are dependent on extreme values
# in that case using median value would be useful
# =============================================================================

descriptive_stats=(df.describe())


### imputation of Age using mean

print("Mean Age=",df["Age"].mean())

df["Age"].fillna(df["Age"].mean(),inplace=True)

### imputing KM data with median value
### because the mean is effeted by the extreme value 
### mean value is higher than median 


print("Median KM=",df["KM"].median())

df["KM"].fillna(df["KM"].median(),inplace=True)

## replacing nan values in hp with mean hp values

print("Mean HP=",df["HP"].mean())

df["HP"].fillna(df["HP"].mean(),inplace=True)



### To  replace the NAN values in catagorical variables we have to
### replace it with the mode or catagory with the highest frequency

## imputing Fuel type missing values

print("Mode of FuelType=",df["FuelType"].value_counts())

## value counts returns a pandas series we can accessteh data using index
## index 0 will always have the value with highest frequency 

df["FuelType"].fillna(df["FuelType"].value_counts().index[0],inplace=True)

## we can eigther use value count or .mode() method to calculate mode 
## .mode() returns values with index in case of multiple mode the index presents the modes with high o low

print("alternative method \n",df["MetColor"].fillna(df["MetColor"].value_counts().index[0],inplace=False).head())

df["MetColor"].fillna(df["MetColor"].mode()[0],inplace=True)


#### Lets Confirm that all the variables are imputed successfully

print(df.isna().sum())


## for a data set with lots of variable it is not possible to mannualy impute all the null values so 
## we automate the process using lambda function

df_final=df_pipeline.apply(lambda x:x.fillna(x.mean()
                                             if x.dtype=="float"
                                             else x.fillna(x.value_counts().index[0])))
## Les check for the success 

df_final.isna().sum()