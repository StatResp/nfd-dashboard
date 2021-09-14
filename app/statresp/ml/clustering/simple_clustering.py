"""
@Author - Sayyed Mohsen Vazirizade
feature space and merges clusters up to a pre-specified number of clusters
"""

#from clustering.base import Clustering_Algo
import pandas as pd
#from forecasters import poisson_reg_forecaster
from sklearn.cluster import KMeans
import numpy as np


class Simple_Cluster:
    def __init__(self,model_name,n_clusters):
        self.model_name = model_name
        self.n_clusters = n_clusters
        #self.name = "Simple Clustering"
        if (self.model_name == 'KM'):
            model = KMeans(n_clusters=n_clusters, random_state=0)
        else:
            model=np.nan
            
            
        self.modelparam=model
        
            
    
        
    def fit(self, df_cluster, metadata=None):
        #fit clustering model to data
        #@param df_train: dataframe which used to count the number of accidents for each segment and use it for clustering training
        #@param df_test and df_predict: dataframe we want to apply clustering on it 
        #@param metadata: 
        #@return: a list indicates the cluster id
        #df_cluster = df_train[[metadata['unit_name'], metadata['pred_name_TF']]].groupby(metadata['unit_name']).mean().reset_index()
        df_cluster['extra']=0  #since KMeans needs at least 2 dimensions, we add a fake dimension.
        self.modelparam = self.modelparam.fit(df_cluster[['extra',metadata['pred_name_TF']]])        
        df_cluster['cluster_label']=self.modelparam.predict(df_cluster[['extra',metadata['pred_name_TF']]])
        self.df_cluster=df_cluster
        self.unit_name=metadata['unit_name']
    
        
    def pred(self,df):
        df_cluster=self.df_cluster
        Max=df_cluster['cluster_label'].value_counts().sort_values().index[0]
        df= pd.merge(df, df_cluster[['cluster_label',self.unit_name]], left_on=self.unit_name, right_on=self.unit_name, how='left')    
        df['cluster_label']=df['cluster_label'].fillna(Max)
        df['cluster_label']=df['cluster_label'].astype(int)
        df['cluster_label']=df['cluster_label']
        
        return df


'''
df_cluster=DF_Regression[['XDSegID','Total_Number_Incidents_average_per_seg','Intercept']].drop_duplicates()

from clustering.base import Clustering_Algo
from numpy.linalg import norm
from itertools import combinations
import pandas as pd
#from forecasters import poisson_reg_forecaster
from clustering.utils import *
from copy import deepcopy
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

modelparam = KMeans(n_clusters=5, random_state=0)
modelparam = modelparam.fit(df_cluster[['Intercept','Total_Number_Incidents_average_per_seg']])
df_cluster['cluster_label']=modelparam.predict(df_cluster[['Intercept','Total_Number_Incidents_average_per_seg']])
print(df_cluster.groupby('cluster_label').count())
print(df_cluster.groupby('cluster_label').max())    
print(df_cluster.groupby('cluster_label').mean())    
print(df_cluster.groupby('cluster_label').min())       


df_cluster['Total_Number_Incidents_average_per_seg'].hist(bins = 50)
for i in df_cluster['cluster_label'].unique().tolist():
    plt.axvline(x=df_cluster[df_cluster['cluster_label']==i]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')
    
    
    
plt.axvline(x=df_cluster[df_cluster['cluster_label']==1]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')
plt.axvline(x=df_cluster[df_cluster['cluster_label']==2]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')    
plt.axvline(x=df_cluster[df_cluster['cluster_label']==3]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')    
plt.axvline(x=df_cluster[df_cluster['cluster_label']==4]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')    
'''