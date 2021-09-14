# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:34:55 2021

@Author - Sayyed Mohsen Vazirizade  s.m.vazirizade@vanderbilt.edu
Parent File for all regression models for incident prediction.
Currently supports -
1. Poisson Regression (Count Based)
2. Negative Binomial Regression (Count Based)
3. Parametric Survival Regression (Time Based)
"""
#from forecasters.non_linear_forecaster import SVM

from statresp.ml.clustering.base import Clustering_Algo
from statresp.ml.resampling.Resampling import Resampling_Algo
from statresp.dataprep.traintest.traintest import create_train_test_dataset, create_predict_dataset, Train_Verification_Split
import numpy as np
import os
import pickle

def train_verif_test_dataset_generator(df_merged,metadata):
    '''
    This function creates the train, valication(verification), test, and prediction datasets.
    It uses the type of the model, the type of clustering, and the type of resampling to create the aforementioned datasets.

    Parameters
    ----------
    df_merged : TYPE
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    dataframe
        train.
    dataframe
        valication(verification).
    dataframe
        test.
    dataframe
        prediction.
    metadata : dictionary
        DESCRIPTION.

    '''
    
    #metadata=add_to_current_metadata(metadata, model_type)
    #print('\n', 'The process for {} has been started:'.format{metadata['current_model']['Name']})
    np.random.seed(metadata['seed_number']) 


    # create train, verification, test, and predictioln
    df_train, df_test= create_train_test_dataset(df_merged, metadata)

    if len(df_train)==0:
        raise ValueError("length of the train dataset is zero")
    if len(df_test)==0:
        raise ValueError("length of the test dataset is zero")

    if metadata['current_model']['Name']!='Naive':
        #clustering
        if metadata['current_model']['cluster_type'] in metadata['available_clustering_methods']:
           Cluster=Clustering_Algo( metadata['current_model']['cluster_type'],metadata['current_model']['cluster_number'])
           Cluster.fit(df_train,metadata)
           df_train= Cluster.pred(df_train,metadata)
           df_test= Cluster.pred(df_test,metadata)
           #print('Summary of total number of segments the average rate of incidents per cluster in the training dataset')
           #print(df_train[['cluster_label', metadata['pred_name_TF']]].groupby('cluster_label').agg({'mean', 'count'}))

           directory=metadata['Output_Address']+'/trained_models/clusters'
           if not os.path.exists(directory):
                os.makedirs(directory)               
           pickle.dump(Cluster, open(directory+'/'+metadata['current_model']['Cluster_Name']+'.sav', 'wb'))
        else:
            #print('No Clustering')
            df_train.loc[:,'cluster_label'] = 0  
            df_test.loc[:,'cluster_label'] = 0  

        
        #train and verification split    
        df_train,df_verif = Train_Verification_Split(df_train,metadata)    
            
        #Resampling    
        if metadata['current_model']['resampling_type'] in metadata['available_resampling_methods']:    
            Resamples=Resampling_Algo(metadata['current_model']['resampling_type'],metadata['current_model']['resampling_rate'] )
            Resamples.Balance_Adjuster(df_train,metadata)   
            #df_train=Resamples.fit_resample(df_train,metadata)
            df_train=Resamples.fit_resample(df_train,metadata)
        #print('Clustering and Resampling Summary in the training dataset: ')
        #print(df_train[[metadata['pred_name_TF'],'cluster_label']].value_counts([metadata['pred_name_TF'],'cluster_label']))

    else:#Naive
        #print('No Clustering')
        df_train.loc[:,'cluster_label'] = 0  
        df_test.loc[:,'cluster_label'] = 0  
        df_train,df_verif = Train_Verification_Split(df_train,metadata)    
    
    
    
    #Savedata(df_train=df_train, df_verif=df_verif,df_test=df_test, Name=metadata['current_model']['Data_Name'],directory=metadata['Output_Address'])
    return df_train,df_verif,df_test

