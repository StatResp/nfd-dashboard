# -*- coding: utf-8 -*-
"""
@Author - Sayyed Mohsen Vazirizade  s.m.vazirizade@vanderbilt.edu
This file includes all the functions can be used to check the validity of the input configuration. 
"""

import pandas as pd
from statresp.ml.forecasters.utils import Correctness_caluclator,Conf_Mat, Concat_AllTestTime, Mean_of_AllTestTime_Results, Correlation_Function, Add_to_Dic, Metric_calculator_per_time, Results_Generator,Naive_adder
import numpy as np



def add_to_metadata(metadata,df_merged=None):
    '''
    This function adds the unique roadway segments to the metadata if df_merged exists.
    Also, if the Naive tage is True, it adds the Naive model to the model list

    Parameters
    ----------
    metadata : dictionary
        DESCRIPTION.
    df_merged : dataframe
        The frame used to extract the unique roadway segments

    Returns
    -------
    metadata : TYPE
        DESCRIPTION.

    '''
    if not df_merged is None:
        metadata['all_segments'] = np.sort(df_merged[metadata['unit_name']].unique()) 
        
    if metadata['naive']==True:    
        metadata['model_type'].append('Naive')
    return metadata

#%%

def Moving_Window_Function(metadata):
    '''
    User can choose rolling window for training by using moving_window_month or moving_window_week. This function is used to adjust the train and test ranges.

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    Window_TrainTest : list
        list of ranges: start_time_train, end_time_train, start_time_test, end_time_test 

    '''
    
    Window_TrainTest=[]
    Offset=pd.DateOffset(hours=metadata['window_size']/3600)
    #case 1
    if metadata['train_test_type']=='moving_window_month':
        Delta=((metadata['end_time_test_abs']-Offset).year               
              -(metadata['start_time_test_abs']).year)*12 + ((metadata['end_time_test_abs'] -  Offset).month           
              -(metadata['start_time_test_abs']).month)
        if Delta==0:
            Window_TrainTest=[[metadata['start_time_train_abs'], metadata['end_time_train_abs'],metadata['start_time_test_abs'], metadata['end_time_test_abs']]]
        else:
            for i in range(Delta+1):
                Window_TrainTest.append([metadata['start_time_train_abs'], metadata['end_time_train_abs']+pd.DateOffset(months=i),metadata['start_time_test_abs']+pd.DateOffset(months=i), metadata['end_time_test_abs']+pd.DateOffset(months=i)])
                #Window_Test.append((metadata['start_time_test']+pd.DateOffset(months=i)))
                
    #case 2
    elif metadata['train_test_type']=='moving_window_week':
        str(metadata['start_time_test_abs']+pd.DateOffset(months=1))
        Delta=np.floor((metadata['end_time_test_abs'] - Offset - metadata['start_time_test_abs']).days/7)
        if Delta==0:
            Window_TrainTest=[[metadata['start_time_train_abs'], metadata['end_time_train_abs'],metadata['start_time_test_abs'], metadata['end_time_test_abs']]]
        else:
            for i in range(Delta+1):
                Window_TrainTest.append([metadata['start_time_train_abs'], metadata['end_time_train_abs']+pd.DateOffset(days=i*7),metadata['start_time_test_abs']+pd.DateOffset(days=i*7), metadata['end_time_test_abs']+pd.DateOffset(days=i*7)])
        
    #case 3
    else:
        Window_TrainTest=[[metadata['start_time_train_abs'], metadata['end_time_train_abs'],metadata['start_time_test_abs'], metadata['end_time_test_abs']]]

    return Window_TrainTest

#%%




def add_to_current_metadata(metadata,model_num, model_type):
    '''
    Since we have various models and training/testing types, each one can if different features.
    This fucntion creates a metadata['current'] and defines the required features.

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.
    model_num : int
        the order of the model.     
    model_type : str
        type of the machine learning method. For now, it can be LR, NN, RF


    Returns
    -------
    metadata : TYPE
        DESCRIPTION.
    model_i : TYPE
        DESCRIPTION.

    '''
    if model_type=='Naive':
        model_i='Naive'
        
        data_i=str(metadata['start_time_train'])+'to'+str(metadata['end_time_test'])
        data_i=data_i.replace(' ','-').replace(':','')
        model_full_name = metadata['model_type'][model_num] + '+' + data_i
        metadata['current_model']= {'Name':model_i , 'Model_Full_Name':model_full_name, 'Data_Name':data_i,'Cluster_Name':None,'model_type': 'Naive',  }
    
    else:
        cluster_i=metadata['cluster_type'][model_num]+str(metadata['cluster_number'][model_num])
        data_i=metadata['resampling_type'][model_num]+str(metadata['resampling_rate'][model_num])+'+'+cluster_i
        model_i=metadata['model_type'][model_num]+'+'+data_i

        cluster_i=cluster_i+'-'+str(metadata['start_time_train'])+'to'+str(metadata['end_time_test'])
        cluster_i=cluster_i.replace(' ','-').replace(':','')        
        
        data_i=data_i+'-'+str(metadata['start_time_train'])+'to'+str(metadata['end_time_test'])
        data_i=data_i.replace(' ','-').replace(':','')
        model_full_name = metadata['model_type'][model_num] + '+' + data_i

        metadata['current_model']= {'Name':model_i, 'Model_Full_Name':model_full_name,'Data_Name':data_i, 'Cluster_Name':cluster_i  , 'model_type': metadata['model_type'][model_num], 'resampling_type':metadata['resampling_type'][model_num],'resampling_rate': metadata['resampling_rate'][model_num], 'cluster_type': metadata['cluster_type'][model_num], 'cluster_number': metadata['cluster_number'][model_num], 'model_hyperparam_address': metadata['model_hyperparam_address'][model_num]}
    
    
    metadata['current_model']['features'] = (metadata['features']).copy()      #feature_ALL incorporates 
    return metadata


def add_to_metadata_prediction(metadata,model,Window_Number_i):
        learn_results={}
        #model_i=('.').join(metadata['model_to_predict'].split('.')[0:-1])
        current_model={}
        current_model['model_type']=model.model_type
        metadata['current_model']=current_model['model_type']
        learn_results[Window_Number_i]={}
        learn_results[Window_Number_i][model.model_type]={}
        learn_results[Window_Number_i][model.model_type]['model']=model
        
        return learn_results,metadata

def Resuluts_Combiner(learn_results,metadata,Type='df_test'):       
    '''
    This function do the following tasks: 
        1) concat predicted values from various windows (if rolling window is selected for training and test) and create DF_Test_spacetime DF
        2) bring all perfromance results in one dataframe as well as adding new pefroamce metrics called perforance 
        3) it also adds the added metrics to learn_results to be sent to html file generator later

    Parameters
    ----------
    learn_results : TYPE
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.
    Type : TYPE, optional
        DESCRIPTION. The default is 'df_test'.

    Returns
    -------
    DF_Test_metric_time : dataframe
        For each time window, the associated performance metrics are reported.
    DF_performance  : dataframe
        The total performance metrics are reported. If rolling window is selected for training and test, it includes the pefroamnce for rolling window seperately.
    DF_Test_spacetime : dataframe
        It includes all of the predicted data.  If rolling window is selected for training and test, the prediction for different rollowing windows are appeneded to each other. 
    model_set : list
        a list of the name of the unique models trained.    

    '''     
    #1)
    DF_Test_spacetime=Concat_AllTestTime(learn_results,metadata,Type=Type)
    #2)DF_performance : Summary of the perforance of the models
    DF_performance  =Mean_of_AllTestTime_Results(learn_results)
    DF_performance = Correlation_Function(DF_Test_spacetime,DF_performance ,metadata)
    #DF_Test_metric_time: accuracy, preciion, F1, recall for each model for test data for each time
    #3)
    DF_Test_metric_time=Metric_calculator_per_time(DF_Test_spacetime,DF_performance ,metadata) #it also adds the metrics for naive model
    #learn_results: this is sent to html file
    learn_results=Add_to_Dic(learn_results,DF_performance )        
    
    DF_Test_metric_time,DF_performance =Correctness_caluclator(DF_Test_spacetime,DF_Test_metric_time,DF_performance , learn_results,metadata, Type=int(1) )
    
    model_set=learn_results[list(learn_results.keys())[0]].keys()

    return DF_Test_metric_time,DF_performance ,DF_Test_spacetime,model_set