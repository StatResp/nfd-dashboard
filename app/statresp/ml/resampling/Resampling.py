# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:26:26 2020

@author: Sayyed Mohsen Vazirizade
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from copy import deepcopy
from scipy.special import factorial
from scipy import stats
from scipy.special import gamma
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE







class Resampling_Algo:

    def __init__(self, sampling_name,rate):
        self.sampling_name=sampling_name
        self.rate=rate

    def Balance_Adjuster(self,df,metadata):
        """
        caclulate the Balance ratio between each cluster for resampling
        @param df: the big regression dataframe
        @param metadata:
        @return BalanceFactor: it is a list of balanced ratio for each cluster sorted by cluster number
        """
        Sparsity=df[['cluster_label',metadata['pred_name_TF']]].groupby('cluster_label').mean().reset_index().sort_values('cluster_label',ascending=True )#[metadata['pred_name_TF']].tolist()
        Sparsity['BalanceFactor']=Sparsity[metadata['pred_name_TF']]/max(Sparsity[metadata['pred_name_TF']])*self.rate
        BalanceFactor=dict()
        for i in range(len(Sparsity)):
            BalanceFactor[Sparsity['cluster_label'].iloc[i]]=Sparsity['BalanceFactor'].iloc[i]
        #print(BalanceFactor)
        self.BalanceFactor=BalanceFactor
        
        

    def Resampling_Func(self,BalanceFactor):
        """
        choose the resampling method
        @param sampling_name: string that explains the resampling method
        @param BalanceFactor: the BalanceFactor for one indivudual cluster
        @return sampling:it is the model for resampling
        """   
        
        if self.sampling_name in ['RUS','ROS','SMOTE']:
            if self.sampling_name=='RUS':        
                
                sampling = RandomUnderSampler(sampling_strategy=BalanceFactor,random_state=1)
            elif self.sampling_name=='ROS':        
                
                sampling = RandomOverSampler(sampling_strategy=BalanceFactor,random_state=1)
            elif self.sampling_name=='SMOTE':
    
                #from imblearn.over_sampling import BorderlineSMOTE
                self.sampling = SMOTE(sampling_strategy=BalanceFactor,random_state=1)
                #oversample = BorderlineSMOTE()
        return sampling
    

    def fit_resample(self, df,metadata):
        df=df.reset_index().rename(columns={'index':'ID'})
        df_=df[['cluster_label','ID',metadata['pred_name_TF']]]
        df_resampled=pd.DataFrame()
        #df_verif_resampled=pd.DataFrame()
        for temp_cluster in self.BalanceFactor.keys():
            #print('cluster number: ',temp_cluster)
            np.random.seed(seed=0)     
            df_r=df_[df_['cluster_label']==temp_cluster]
            #df_r,df_verif = Train_Verification_Split(df_r,metadata)    
            print('Cluster:', temp_cluster, ', Balance Factor:', self.BalanceFactor[temp_cluster])
            sampling= self.Resampling_Func(self.BalanceFactor[temp_cluster] )
            df_r,_ = sampling.fit_resample(df_r,df_r[metadata['pred_name_TF']])
            df_resampled=df_resampled.append(df_r)
            #df_verif_resampled=df_verif_resampled.append(df_verif)
            #df_[['Total_Number_Incidents_TF','cluster_label']].value_counts(['Total_Number_Incidents_TF','cluster_label'])
        
        df_resampled=df_resampled.reset_index().drop(['cluster_label','index',metadata['pred_name_TF']],axis=1)
        df_resampled=pd.merge(df_resampled, df, left_on='ID', right_on='ID', how='left')
        df_resampled=df_resampled.drop('ID',axis=1)
        
        #df_verif_resampled=df_verif_resampled.reset_index().drop(['cluster_label','index',metadata['pred_name_TF']],axis=1)
        #df_verif_resampled=pd.merge(df_verif_resampled, df, left_on='ID', right_on='ID', how='left')
        #df_verif_resampled=df_verif_resampled.drop('ID',axis=1)
                
        
        return df_resampled #,df_verif_resampled
    
    
