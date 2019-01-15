#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import datetime
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neural_network
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import glob
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold


# In[31]:


#Loading CSV File and converting it into pandas dataframe
table = pd.read_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Input\\dataset.csv',index_col=0)
df1 = pd.DataFrame(table)
#Get only statewise total data
df2=df1.loc[df1['DISTRICT'] == 'TOTAL']
#print(df2)


# In[32]:


#Feature Selection: Drop the not required features
df2=df2.drop(['DISTRICT','CUSTODIAL RAPE','OTHER RAPE','KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS','KIDNAPPING AND ABDUCTION OF OTHERS','AUTO THEFT','OTHER THEFT'],1)


# In[33]:


#Saving the cleaned dataset
df2.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\clean_dataset.csv')


# In[34]:


#Get different dataframes for each year
df2011 = df2.loc[df2['YEAR'] == 2011]
df2001 = df2.loc[df2['YEAR'] == 2001]
df2002 = df2.loc[df2['YEAR'] == 2002]
df2003 = df2.loc[df2['YEAR'] == 2003]
df2004 = df2.loc[df2['YEAR'] == 2004]
df2005 = df2.loc[df2['YEAR'] == 2005]
df2006 = df2.loc[df2['YEAR'] == 2006]
df2007 = df2.loc[df2['YEAR'] == 2007]
df2008 = df2.loc[df2['YEAR'] == 2008]
df2009 = df2.loc[df2['YEAR'] == 2009]
df2010 = df2.loc[df2['YEAR'] == 2011]
df2012 = df2.loc[df2['YEAR'] == 2012]


# In[35]:


#Dropping the column YEAR from each dataframe
df2011=df2011.drop(['YEAR'],1)
df2001=df2001.drop(['YEAR'],1)
df2002=df2002.drop(['YEAR'],1)
df2003=df2003.drop(['YEAR'],1)
df2004=df2004.drop(['YEAR'],1)
df2005=df2005.drop(['YEAR'],1)
df2006=df2006.drop(['YEAR'],1)
df2007=df2007.drop(['YEAR'],1)
df2008=df2008.drop(['YEAR'],1)
df2009=df2009.drop(['YEAR'],1)
df2010=df2010.drop(['YEAR'],1)
df2012=df2012.drop(['YEAR'],1)


# In[36]:


#Obtaining the crime rates for different states for different years 
df2001_dividedByTotalIPC=df2001.divide(df2001['TOTAL IPC CRIMES'],axis=0)*100000
df2002_dividedByTotalIPC=df2002.divide(df2002['TOTAL IPC CRIMES'],axis=0)*100000
df2003_dividedByTotalIPC=df2003.divide(df2003['TOTAL IPC CRIMES'],axis=0)*100000
df2004_dividedByTotalIPC=df2004.divide(df2004['TOTAL IPC CRIMES'],axis=0)*100000
df2005_dividedByTotalIPC=df2005.divide(df2005['TOTAL IPC CRIMES'],axis=0)*100000
df2006_dividedByTotalIPC=df2006.divide(df2006['TOTAL IPC CRIMES'],axis=0)*100000
df2007_dividedByTotalIPC=df2007.divide(df2007['TOTAL IPC CRIMES'],axis=0)*100000
df2008_dividedByTotalIPC=df2008.divide(df2008['TOTAL IPC CRIMES'],axis=0)*100000
df2009_dividedByTotalIPC=df2009.divide(df2009['TOTAL IPC CRIMES'],axis=0)*100000
df2010_dividedByTotalIPC=df2010.divide(df2010['TOTAL IPC CRIMES'],axis=0)*100000
df2011_dividedByTotalIPC=df2011.divide(df2011['TOTAL IPC CRIMES'],axis=0)*100000
df2012_dividedByTotalIPC=df2012.divide(df2012['TOTAL IPC CRIMES'],axis=0)*100000


# In[37]:


#Dropping the column 'Total IPC Crimes' since it is not needed
df2001_droppedTotalIPC=df2001_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2002_droppedTotalIPC=df2002_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2003_droppedTotalIPC=df2003_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2004_droppedTotalIPC=df2004_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2005_droppedTotalIPC=df2005_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2006_droppedTotalIPC=df2006_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2007_droppedTotalIPC=df2007_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2008_droppedTotalIPC=df2008_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2009_droppedTotalIPC=df2009_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2010_droppedTotalIPC=df2010_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2011_droppedTotalIPC=df2011_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)
df2012_droppedTotalIPC=df2012_dividedByTotalIPC.drop(['TOTAL IPC CRIMES'],1)


# In[7]:


#Min-max scaling the data
scaler = MinMaxScaler()
normalised_data_2001=scaler.fit_transform(df2001_droppedTotalIPC)
normalised_data_2002=scaler.fit_transform(df2002_droppedTotalIPC)
normalised_data_2003=scaler.fit_transform(df2003_droppedTotalIPC)
normalised_data_2004=scaler.fit_transform(df2004_droppedTotalIPC)
normalised_data_2005=scaler.fit_transform(df2005_droppedTotalIPC)
normalised_data_2006=scaler.fit_transform(df2006_droppedTotalIPC)
normalised_data_2007=scaler.fit_transform(df2007_droppedTotalIPC)
normalised_data_2008=scaler.fit_transform(df2008_droppedTotalIPC)
normalised_data_2009=scaler.fit_transform(df2009_droppedTotalIPC)
normalised_data_2010=scaler.fit_transform(df2010_droppedTotalIPC)
normalised_data_2011=scaler.fit_transform(df2011_droppedTotalIPC)
normalised_data_2012=scaler.fit_transform(df2012_droppedTotalIPC)


# In[8]:


#normalised_data_2011


# In[38]:


#converting the numpy array to dataframes
df_normalized_data_2001= pd.DataFrame(normalised_data_2001)
df_normalized_data_2002= pd.DataFrame(normalised_data_2002)
df_normalized_data_2003= pd.DataFrame(normalised_data_2003)
df_normalized_data_2004= pd.DataFrame(normalised_data_2004)
df_normalized_data_2005= pd.DataFrame(normalised_data_2005)
df_normalized_data_2006= pd.DataFrame(normalised_data_2006)
df_normalized_data_2007= pd.DataFrame(normalised_data_2007)
df_normalized_data_2008= pd.DataFrame(normalised_data_2008)
df_normalized_data_2009= pd.DataFrame(normalised_data_2009)
df_normalized_data_2010= pd.DataFrame(normalised_data_2010)
df_normalized_data_2011= pd.DataFrame(normalised_data_2011)
df_normalized_data_2012= pd.DataFrame(normalised_data_2012)


# In[57]:


df_normalized_data_2001.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2001_dividedbytotalIPC.csv')
df_normalized_data_2002.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2002_dividedbytotalIPC.csv')
df_normalized_data_2003.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2003_dividedbytotalIPC.csv')
df_normalized_data_2004.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2004_dividedbytotalIPC.csv')
df_normalized_data_2005.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2005_dividedbytotalIPC.csv')
df_normalized_data_2006.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2006_dividedbytotalIPC.csv')
df_normalized_data_2007.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2007_dividedbytotalIPC.csv')
df_normalized_data_2008.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2008_dividedbytotalIPC.csv')
df_normalized_data_2009.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2009_dividedbytotalIPC.csv')
df_normalized_data_2010.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2010_dividedbytotalIPC.csv')
df_normalized_data_2011.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2011_dividedbytotalIPC.csv')
df_normalized_data_2012.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Normalised_Datasets\\normalised_dataset_2012_dividedbytotalIPC.csv')


# In[9]:


#Retaining the column/index names after normalisation
df_normalized_data_2001.columns = df2001_droppedTotalIPC.columns
df_normalized_data_2001.index=df2001_droppedTotalIPC.index
df_normalized_data_2002.columns = df2002_droppedTotalIPC.columns
df_normalized_data_2002.index=df2002_droppedTotalIPC.index
df_normalized_data_2003.columns = df2003_droppedTotalIPC.columns
df_normalized_data_2003.index=df2003_droppedTotalIPC.index
df_normalized_data_2004.columns = df2004_droppedTotalIPC.columns
df_normalized_data_2004.index=df2004_droppedTotalIPC.index
df_normalized_data_2005.columns = df2005_droppedTotalIPC.columns
df_normalized_data_2005.index=df2005_droppedTotalIPC.index
df_normalized_data_2006.columns = df2006_droppedTotalIPC.columns
df_normalized_data_2006.index=df2006_droppedTotalIPC.index
df_normalized_data_2007.columns = df2007_droppedTotalIPC.columns
df_normalized_data_2007.index=df2007_droppedTotalIPC.index
df_normalized_data_2008.columns = df2008_droppedTotalIPC.columns
df_normalized_data_2008.index=df2008_droppedTotalIPC.index
df_normalized_data_2009.columns = df2009_droppedTotalIPC.columns
df_normalized_data_2009.index=df2009_droppedTotalIPC.index
df_normalized_data_2010.columns = df2010_droppedTotalIPC.columns
df_normalized_data_2010.index=df2010_droppedTotalIPC.index
df_normalized_data_2011.columns = df2011_droppedTotalIPC.columns
df_normalized_data_2011.index=df2011_droppedTotalIPC.index
df_normalized_data_2012.columns = df2012_droppedTotalIPC.columns
df_normalized_data_2012.index=df2012_droppedTotalIPC.index


# In[10]:


#Defining the weights for the different crimes
weights = [4,3,2,3,2,2,2,2,1,1,1,1,1,3,2,2,3,1,1,1,2,1,2]


# In[11]:


#Getting the new dataframes with the weighted features
df_normalized_data_2001_Weighted=df_normalized_data_2001.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2002_Weighted=df_normalized_data_2002.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2003_Weighted=df_normalized_data_2003.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2004_Weighted=df_normalized_data_2004.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2005_Weighted=df_normalized_data_2005.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2006_Weighted=df_normalized_data_2006.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2007_Weighted=df_normalized_data_2007.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2008_Weighted=df_normalized_data_2008.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2009_Weighted=df_normalized_data_2009.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2010_Weighted=df_normalized_data_2010.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2011_Weighted=df_normalized_data_2011.mul(pd.Series(weights).values, axis=1)
df_normalized_data_2012_Weighted=df_normalized_data_2012.mul(pd.Series(weights).values, axis=1)


# In[20]:


#Storing the transformed data set into csv files
df_normalized_data_2001_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2001_W_4_3_2_1.csv')
df_normalized_data_2002_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2002_W_4_3_2_1.csv')
df_normalized_data_2003_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2003_W_4_3_2_1.csv')
df_normalized_data_2004_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2004_W_4_3_2_1.csv')
df_normalized_data_2005_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2005_W_4_3_2_1.csv')
df_normalized_data_2006_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2006_W_4_3_2_1.csv')
df_normalized_data_2007_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2007_W_4_3_2_1.csv')
df_normalized_data_2008_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2008_W_4_3_2_1.csv')
df_normalized_data_2009_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2009_W_4_3_2_1.csv')
df_normalized_data_2010_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2010_W_4_3_2_1.csv')
df_normalized_data_2011_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\weighted_dataset_2011_W_4_3_2_1.csv')
df_normalized_data_2012_Weighted.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\Weighted_Datasets_2012\\weighted_dataset_2012_W_4_3_2_1.csv')


# In[12]:


#Clustering using K-Means with K=3
mat2001 = df_normalized_data_2001_Weighted.values
mat2002 = df_normalized_data_2002_Weighted.values
mat2003 = df_normalized_data_2003_Weighted.values
mat2004 = df_normalized_data_2004_Weighted.values
mat2005 = df_normalized_data_2005_Weighted.values
mat2006 = df_normalized_data_2006_Weighted.values
mat2007 = df_normalized_data_2007_Weighted.values
mat2008 = df_normalized_data_2008_Weighted.values
mat2009 = df_normalized_data_2009_Weighted.values
mat2010 = df_normalized_data_2010_Weighted.values
mat2011 = df_normalized_data_2011_Weighted.values
mat2012 = df_normalized_data_2012_Weighted.values
km2001 = KMeans(n_clusters=3,random_state=0)
km2001.max_iter=500
km2001.fit(mat2001)
labels2001 = km2001.labels_
results2001 = pd.DataFrame([df_normalized_data_2001_Weighted.index,labels2001]).T
km2002 = KMeans(n_clusters=3,random_state=0)
km2002.max_iter=500
km2002.fit(mat2002)
labels2002 = km2002.labels_
results2002 = pd.DataFrame([df_normalized_data_2002_Weighted.index,labels2002]).T
km2003 = KMeans(n_clusters=3,random_state=0)
km2003.max_iter=500
km2003.fit(mat2003)
labels2003 = km2003.labels_
results2003 = pd.DataFrame([df_normalized_data_2003_Weighted.index,labels2003]).T
km2004 = KMeans(n_clusters=3,random_state=0)
km2004.max_iter=500
km2004.fit(mat2004)
labels2004 = km2004.labels_
results2004 = pd.DataFrame([df_normalized_data_2004_Weighted.index,labels2004]).T
km2005 = KMeans(n_clusters=3,random_state=0)
km2005.max_iter=500
km2005.fit(mat2005)
labels2005 = km2005.labels_
results2005 = pd.DataFrame([df_normalized_data_2005_Weighted.index,labels2005]).T
km2006 = KMeans(n_clusters=3,random_state=0)
km2006.max_iter=500
km2006.fit(mat2006)
labels2006 = km2006.labels_
results2006 = pd.DataFrame([df_normalized_data_2006_Weighted.index,labels2006]).T
km2007 = KMeans(n_clusters=3,random_state=0)
km2007.max_iter=500
km2007.fit(mat2007)
labels2007 = km2007.labels_
results2007 = pd.DataFrame([df_normalized_data_2007_Weighted.index,labels2007]).T
km2008 = KMeans(n_clusters=3,random_state=0)
km2008.max_iter=500
km2008.fit(mat2008)
labels2008= km2008.labels_
results2008 = pd.DataFrame([df_normalized_data_2008_Weighted.index,labels2008]).T
km2009 = KMeans(n_clusters=3,random_state=0)
km2009.max_iter=500
km2009.fit(mat2009)
labels2009 = km2009.labels_
results2009 = pd.DataFrame([df_normalized_data_2009_Weighted.index,labels2009]).T
km2010 = KMeans(n_clusters=3,random_state=0)
km2010.max_iter=500
km2010.fit(mat2010)
labels2010 = km2010.labels_
results2010 = pd.DataFrame([df_normalized_data_2010_Weighted.index,labels2010]).T
km2011 = KMeans(n_clusters=3,random_state=0)
km2011.max_iter=500
km2011.fit(mat2011)
labels2011 = km2011.labels_
results2011 = pd.DataFrame([df_normalized_data_2011_Weighted.index,labels2011]).T
km2012 = KMeans(n_clusters=3,random_state=0)
km2012.max_iter=500
km2012.fit(mat2012)
labels2012 = km2012.labels_
results2012 = pd.DataFrame([df_normalized_data_2012_Weighted.index,labels2012]).T


# In[21]:


#Cluster Validation using Silhouette Co-efficients for 2001

silhouette_avg=[]
silhouette_avg1 = silhouette_score(df_normalized_data_2001_Weighted, labels2001)
#silhouette_avg
print("Silhouette co-efficient 2001: "+str(silhouette_avg1))
silhouette_avg.append(silhouette_avg1)

#Cluster Validation using Silhouette Co-efficients for 2002
silhouette_avg2 = silhouette_score(df_normalized_data_2002_Weighted, labels2002)
#silhouette_avg
print("Silhouette co-efficient 2002: "+str(silhouette_avg2))
silhouette_avg.append(silhouette_avg2)

#Cluster Validation using Silhouette Co-efficients for 2003
silhouette_avg3 = silhouette_score(df_normalized_data_2003_Weighted, labels2003)
#silhouette_avg
print("Silhouette co-efficient 2003: "+str(silhouette_avg3))
silhouette_avg.append(silhouette_avg3)

#Cluster Validation using Silhouette Co-efficients for 2004
silhouette_avg4 = silhouette_score(df_normalized_data_2004_Weighted, labels2004)
#silhouette_avg
print("Silhouette co-efficient 2004: "+str(silhouette_avg4))
silhouette_avg.append(silhouette_avg4)

#Cluster Validation using Silhouette Co-efficients for 2005
silhouette_avg5 = silhouette_score(df_normalized_data_2005_Weighted, labels2005)
#silhouette_avg
print("Silhouette co-efficient 2005: "+str(silhouette_avg5))
silhouette_avg.append(silhouette_avg5)

#Cluster Validation using Silhouette Co-efficients for 2006
silhouette_avg6 = silhouette_score(df_normalized_data_2006_Weighted, labels2006)
#silhouette_avg
print("Silhouette co-efficient 2006: "+str(silhouette_avg6))
silhouette_avg.append(silhouette_avg6)

#Cluster Validation using Silhouette Co-efficients for 2007
silhouette_avg7 = silhouette_score(df_normalized_data_2007_Weighted, labels2007)
#silhouette_avg
print("Silhouette co-efficient 2007: "+str(silhouette_avg7))
silhouette_avg.append(silhouette_avg7)

#Cluster Validation using Silhouette Co-efficients for 2008
silhouette_avg8 = silhouette_score(df_normalized_data_2008_Weighted, labels2008)
#silhouette_avg
print("Silhouette co-efficient 2008: "+str(silhouette_avg8))
silhouette_avg.append(silhouette_avg8)

#Cluster Validation using Silhouette Co-efficients for 2009
silhouette_avg9 = silhouette_score(df_normalized_data_2009_Weighted, labels2009)
#silhouette_avg
print("Silhouette co-efficient 2009: "+str(silhouette_avg9))
silhouette_avg.append(silhouette_avg9)

#Cluster Validation using Silhouette Co-efficients for 2010
silhouette_avg10 = silhouette_score(df_normalized_data_2010_Weighted, labels2010)
#silhouette_avg
print("Silhouette co-efficient 2010:  "+str(silhouette_avg10))
silhouette_avg.append(silhouette_avg10)

#Cluster Validation using Silhouette Co-efficients for 2011
silhouette_avg11 = silhouette_score(df_normalized_data_2011_Weighted, labels2011)
#silhouette_avg
print("Silhouette co-efficient 2011: "+str(silhouette_avg11))
silhouette_avg.append(silhouette_avg11)

#Cluster Validation using Silhouette Co-efficients for 2012
silhouette_avg12 = silhouette_score(df_normalized_data_2012_Weighted, labels2012)
#silhouette_avg
print("Silhouette co-efficient 2012: "+str(silhouette_avg12))
silhouette_avg.append(silhouette_avg12)

Avg_of_silhouette_avg = sum(silhouette_avg)/len(silhouette_avg)
print("The average Silhouette co-efficient of cluster validation is "+str(Avg_of_silhouette_avg))


# In[23]:


#Storing the resulting cluster labels in csv files
results2001.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2001.csv')
results2002.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2002.csv')
results2003.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2003.csv')
results2004.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2004.csv')
results2005.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2005.csv')
results2006.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2006.csv')
results2007.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2007.csv')
results2008.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2008.csv')
results2009.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2009.csv')
results2010.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2010.csv')
results2011.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\labels2011.csv')
results2012.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\Clustered_Labels\\Clustered_Labels_2012\\labels2012.csv')


# In[63]:


#km2012.cluster_centers_[0]


# In[64]:


#km2012.cluster_centers_[1]


# In[65]:


#km2012.cluster_centers_[2]


# In[6]:


#Creating the training labels as a dataframe for performing the classification
path =r'D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\LabelsAfterAnalysingClusterCentroids2001-2012' 
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)


# In[8]:


#Saving the training data set in a csv file
#frame.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\LabelsAfterClustering2001-2012\\train.csv')


# In[7]:


#Retreiving only the labels and leaving out the names of the states from the training data frame
OnlyLabels=frame.iloc[:,-1]


# In[22]:


#Converting it to a csv file
OnlyLabels.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Input\\InputClassification\\OnlyLabelsAfterClustering2001-2012.csv',index = False)


# In[8]:


#Creating the training dataset as a dataframe for performing the classification
path =r'D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frameDataset = pd.concat(list_).drop(['STATE/UT'],1)
frameDataset.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Input\\InputClassification\\weighted_2001_2011_Merged_Datasets.csv',index = False)


# In[30]:


#Generating the testing dataset as a dataframe for the year 2012
test = pd.read_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Intermediate_Results\\Weighted_Datasets\\Weighted_Datasets_2012\\weighted_dataset_2012_W_4_3_2_1.csv',index_col=0)


# In[68]:

#Classification using kNN with k=1
classifier = KNeighborsClassifier(n_neighbors=1)  
classifier.fit(frameDataset,OnlyLabels)  
#Performing classification for the year 2012
predictedLabel2012 = classifier.predict(test)
predictedLabel2012=pd.DataFrame(predictedLabel2012)
predictedLabel2012.to_csv('D:\IIITD\FirstSem\DMG\project\DMG_MT18048_MT18108_MT18147\\Results\\prediction2012Labels.csv',index = False)

# In[10]:


#Converting the Training datasets which are in numpy array to dataframe
X = frameDataset.values
#Converting the Training Labels to dataframe
Y = OnlyLabels.values
#10-Fold CLuster validation
K=10
kf = KFold(n_splits=K)
#K-Fold splits our training datasets to 10 splits
kf.get_n_splits(X)


# In[22]:


#Running K-Fold on training datasets
#a=[]
i=0
accuracy=[]
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    #Spliting the datasets to train and test
    X_train, X_test = X[train_index], X[test_index]
    #Spliting the labels to train and test
    Y_train, Y_test = Y[train_index], Y[test_index]
    #Classification using kNN with k=1
    classifier = KNeighborsClassifier(n_neighbors=1)  
    classifier.fit(X_train,Y_train)  
    predictedLabelAll = classifier.predict(X_test)
    #a.append(pd.DataFrame(labelAll))
    #For each split calculate accuracy and append it to accuracy array
    acc= (((np.sum(Y_test==predictedLabelAll))/(len(X)/K))*100)
    accuracy.append(acc)
    i=i+1
    print("Fold "+str(i)+" accuracy is "+str(acc) )


# In[23]:


#Calculating the average accuracy of the classifier
avgAccuracy = sum(accuracy)/len(accuracy)
print("The average accuracy of classification is "+str(avgAccuracy))