# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:48:20 2020

@author: Shankha
"""



"""%%%%Data Analytics Framework%%%"""
# =============================================================================
# 1.Poblem Statement ?
# # Predict salary Level to deliver Subsidy
# 2.Problem Coseptualization ?
# Develope  Classifier to Predict the Salary 
# 3.Solution conceptualization?
# Binary classifier
# 4.Method Identification 
#
# 5.Solution Realization 
# Simplify the data sysem by reducing the number of veriables without 
#sacrifising the accuracy
# =============================================================================

##importing libraries and data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#importing train test split for data partition 
from sklearn.model_selection import train_test_split
# importing logistic regression for binary classification
from sklearn.linear_model import LogisticRegression
## importing performance metrices
from sklearn.metrics import accuracy_score,confusion_matrix

sns.set_style("darkgrid")

### Importing the data set
income=pd.read_csv("income.csv")

### copying data to workwith

df=income.copy()
df1=income.copy()

##tocheck the datatype

print(df.info())

## Scanning missing values

print(df.isnull().sum())



##*** We have no missing values in our dataframe

# =============================================================================
# ### BAsic Descriptive statistics of dataframe
# =============================================================================

summery_numerical=df.describe()
summery_categorical=df.describe(include="O")

##** checking for the catagories available under eatch categorical vaeiable

print(pd.unique(df["JobType"]))

print(pd.unique(df["occupation"]))

##** implement unique catagory matrix
Unique={}
for i in  df.select_dtypes("object").columns:
    Unique[i]=pd.unique(df[i])

##** missing data as " ?" found as catagories

df=pd.read_csv("income.csv", na_values=[" ?","??","???"])

##** again scanning for the missing values


print(df.isnull().sum())

null_data=df[df.isnull().any(axis=1)]

# =============================================================================
####################### Conclusion from missing data ##################### 
""" we have 1816 rows of missing in occupation and 1809 rows of missing in JobType
    Two categrical variables have missing data Occupation and JobType
    1816-1809=7 data in JobType more missing because occupation is never worked
"""
# 
# =============================================================================
# Assumption before droping the data 
# #as compared to the total no of data points rows containing null values are less
# #devicing imputation method based on the data willbe very  difficult 
# =============================================================================

df.dropna(axis=0,inplace=True)
print("df after droping missing value",df.isnull().sum())
# =============================================================================
#  Assumption before imputation (filling null values)(not recomended )
# =============================================================================
#if we drop all these coloumns we will be miissing out a lot of data 
# ##Filling missing values
# exploratory data analysis to find out the type of imputation 
# only Jobtype and Occupation hasx the missing data
## as both of them are categorical variables so we will impute mode replacing the missing data 
# =============================================================================


## 
print("job_type=",df["JobType"].mode()[0])
df1["JobType"].fillna(df["JobType"].mode()[0],inplace=True)
df1["occupation"].fillna(df["occupation"].mode()[0],inplace=True)

#** check if the na values are handled succesfully
print("df1 after imputing all values",df1.isnull().sum()) ## Success



"""Exploratory Data Analysis"""


"""Frequency Table , Probability Table and Correlation"""

# =============================================================================
# ## pearson coeef shows linear corelation or linear degree of association but can not indicate anything 
# ## about nonlinearity 
# ## so calculation of pearson as well as spearman can help us conclude the degree of association
# =============================================================================
corelation_spearman=df.corr(method="spearman")
corelation_pearson=df.corr(method="pearson")
#### now with imputed dataset
corelation_spearman1=df1.corr(method="spearman")
corelation_pearson1=df1.corr(method="pearson")
# =============================================================================
# ##** no two variale has strong linear correlation as coef is near to 0
# ##** as for some pair of variable spearman coef is greater than pearson implies there might be some 
# ## non linear association although its very low
# ** comparing coefitient showed that impting missing valus didnt rally impacted much
# =============================================================================

### Frequency distribution of gender 

gender_dist=pd.crosstab(index=df["gender"],columns="count")

"""gender vs salary status """
gender_salarystatus=pd.crosstab(index=df["gender"],columns=df["SalStat"],normalize="index")

#** compared to men more number  of women has salary under 50k
""" joint probability of """

joint_age_SalStat=pd.crosstab(index=df["SalStat"],columns=df["age"],normalize="index")

##$** most people from every age group has salary lesser than 50k
##**all people from  some age group has lalary status lesser than 50k ex
"""** People from age group 44 to 54 has higher probability of salary status more than 50k"""


"""(conditional prob) probability of salarystatus given education type """
salStat_edu=pd.crosstab(index=df["EdType"],columns=df["SalStat"],normalize="index",margins=True )

""" conclusion Bachelor degree ,Hs grads , masters and some college degree holders has"""
"""the highesy probability of getting salary more than 50k"""
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

""" Visualisating the descriptive statistics"""


""" Frequency Distriution of  Salary Status to know the stability of target variable"""
plt.figure(1)
plt.title('freq dist of Salary Status ')
sns.countplot(x="SalStat",data=df)

#** amnongst the sample set maximum datapoints belongs to Salary < 50k ;

""" Frequency Distriution of age group"""
plt.figure(2)
plt.title('freq dist of age ')
plt.xlabel("age")
sns.distplot(df["age"],bins=10, kde=False)

##** People with age group 40-45 has highest frequency

"""Group boxplot to see the relationship between age and salstat """
plt.figure(3)
plt.title(" age vs salary status")
sns.boxplot(x=df["age"],y=df["SalStat"])

"""Salary Status vs catagorical variables"""
##** How Salary status is varing across the job type

plt.figure(4)
sns.countplot(y="JobType",data=df,hue="SalStat")

## TO get how slary stat varries with jobtype we have to create two way table 
#** created earlier
salStat_Job=pd.crosstab(index=df["JobType"],columns=df["SalStat"],normalize="index",margins=True )
salStat_Job=salStat_Job.applymap(lambda x:x*100)
##** maximum people works for  private firm
##  self employed people have the highest probabilirty to earn more than 50k
##** that is why this is a important veriable for classifying salary status
plt.figure(5)
sns.countplot(y="EdType",data=df,hue="SalStat")
salStat_edu=salStat_edu.applymap(lambda x:x*100)
##**
#** maximum people have passed the HS
##doctorate holders have the highest probability to earn >50k
##** that is why this is a important veriable for classifying salary status


"frequency distribution of capital loss and capital gain"

plt.figure(6)

sns.histplot(df["capitalgain"])
##** very few people have invested in the stock so their captal gain is 0 
plt.figure(7)
sns.histplot(x="capitalloss",data=df)
##** Veryfew people didnt eigther inested or they havnt lost anything 

plt.figure(8)
sns.histplot(x="capitalgain",data=df,hue="SalStat")
##** people who have invested and have capital gain have salary >50k

"hours spent per week vs Salary Status"
plt.figure(9)
sns.boxplot(y=df["hoursperweek"],x
            =df["SalStat"],palette="hls")
##** people who have spent 45 hours  on an avarage are more likely to earn >50k

## as the mean of >50K plot and < 50k plot varring a lot and the median values are varring so 
## we can say there is a association  

#%%
# Before moving forward copying the dataset for future use\

df_new=df.copy()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#converting cartagorical target veriable to numerical variable

## method 1 using dictionary maping

df["SalStat"]=df["SalStat"].map({" less than or equal to 50,000":0," greater than 50,000":1})

# using one hot encription 


df=pd.get_dummies(data=df,drop_first=True)

#%%

# =============================================================================
"""                     Logitic Regression         """
#  =============================================================================

"""Seperating the dependent and independent variabl"""

x_data=df.drop("SalStat",axis=1)#feature set

x_data=x_data.values

y_data=df["SalStat"].values


#spliting the dataset into Test set and Train set
X_train,X_test,Y_train,Y_test=train_test_split(x_data,y_data,test_size=0.3,random_state=0)

"""logistic regression classifier"""

## may need to terae through different values to find best fit 

model=LogisticRegression(max_iter=900)

## training the model 

model.fit(X_train,Y_train)

# coeficients 

model.coef_

#interception 

model.intercept_

## prediction of the model using the test data 

y_pred=model.predict(X_test)

### Accuracy metrices from the model 

Accuracy_score=accuracy_score(Y_test,y_pred) 

Confusion_matrix=confusion_matrix(Y_test,y_pred)

#%%


""" misclassified samples """

  
print("misclassified samples = ", (y_pred !=Y_test).sum())

## through exploratory data analysis we  can reduce the feature set droping some insignificant variable

## to improve the model accuracy

# Again Training logistic regression model with the new data with  reduced features set 

##** droping insignificant data from the feature set 
df_new=df_new.drop(["JobType","race","gender","nativecountry"],axis=1)

df_new["SalStat"]=df_new["SalStat"].map({" less than or equal to 50,000":0," greater than 50,000":1})

# using one hot encription 

df_new=pd.get_dummies(df_new,drop_first=True)

X_new_data=df_new.drop("SalStat",axis=1).values


Y_new_data=df["SalStat"].values

## splitinfg the data set

X_train_new,X_test_new,Y_train_new,Y_test_new=train_test_split(X_new_data,Y_new_data,test_size=0.3,random_state=0)

#training the model with new data 


model_new=LogisticRegression(max_iter=900)

## traininig the model 

model_new.fit(X_train_new,Y_train_new)

## model coefficients 


model_new.coef_

## model intersection 

model_new.intercept_

""" Prediction using the test set """

Y_pred_new=model_new.predict(X_test_new)

""" Model Evaluation """
confusion_matrix_new=confusion_matrix(Y_test_new,Y_pred_new)

Accuracy_score_new=accuracy_score(Y_test_new,Y_pred_new)

print(" Accuracy = {} \n Accuacy improvement in % ={}".format(Accuracy_score_new,(Accuracy_score_new-Accuracy_score)*100))

""" Accuracy Score drops alittle but with all those insignificant variables removed model is more robust"""

#%%
"""                   KNN classification                                     """

##** Building knn classifier for the same problem statement with the same dtat set 

from sklearn.neighbors import KNeighborsClassifier

##**KNN Classifier instance with differnt k
Accuracy_score_KNN={}
for i in range(1,20):
    
    KNN=KNeighborsClassifier(n_neighbors=i,metric="minkowski")
    
    ##** Training the Model 
    
    KNN.fit(X_train_new,Y_train_new)
    
    ##** Predicting the valus
    Y_pred_knn=KNN.predict(X_test_new)
    
    ##** Aaccuracy Score 
    
    Accuracy_score_KNN[i]=accuracy_score(Y_test_new,Y_pred_knn)
    