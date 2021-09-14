# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 23:51:43 2021

@Author - Sayyed Mohsen Vazirizade  s.m.vazirizade@vanderbilt.edu
"""

from statresp.ml.forecasters.utils import  update_meta_features
from statresp.getdata.All_DFs import get_future_data
from statresp.utils.filtering.spatial import Spatial_Filtering
from statresp.utils.checkers import Time_checker,feature_checker
import pandas as pd
import pickle
import numpy as np

from statresp.ml.forecasters.utils import dummy_maker

def create_train_test_dataset(df, metadata):
    '''
    This function splits the data into train and test based on the information provided in the config file.

    Parameters
    ----------
    df : dataframe
        the whole dataframe needed to be split into train and test.
    metadata : dictionary
        metadata.

    Returns
    -------
    df_train : dataframe
        df for training.
    df_test : dataframe
        df for test.

    '''
    np.random.seed(metadata['seed_number']) 
        
        
    if (metadata['train_test_type']=='simple') | (metadata['train_test_type']=='moving_window_month') | (metadata['train_test_type']=='moving_window_week'):
        #print('The train and test datasets are defined just based on the input dates')
        df_train   = df.loc[(df[metadata['time_column']] >= metadata['start_time_train'])   &   (df[metadata['time_column']] < metadata['end_time_train']) ]
        df_test   = df.loc[(df[metadata['time_column']] >= metadata['start_time_test'])   &   (df[metadata['time_column']] < metadata['end_time_test']) ]
        
    elif metadata['train_test_type']=='random_pseudo':
        #print('Two weeks are hard-codedly added for genearing test dataset. ')
        df_learn   = df.loc[(df[metadata['time_column']] >= metadata['start_time_train'])   &   (df[metadata['time_column']] < metadata['end_time_train']) ]
        Week_2018=(((df_learn['time_local'].dt.week==19) | (df_learn['time_local'].dt.week==30)) & (df_learn['time_local'].dt.year==2018))
        Week_2019=(((df_learn['time_local'].dt.week==19) | (df_learn['time_local'].dt.week==31)) & (df_learn['time_local'].dt.year==2019))
        mask= (Week_2018 | Week_2019)==False
        df_train = df_learn[mask]
        df_test = df_learn[~mask]

    elif metadata['train_test_type']=='ratio_random':
        # print('Samples are randomly selected from the total range of train and test for test dataset.')
        df_learn   = df.loc[(df[metadata['time_column']] >= metadata['start_time_train'])   &   (df[metadata['time_column']] < metadata['end_time_train']) ]
        split_point = metadata['train_test_split']
        mask=df_learn.index.isin(df_learn.sample(int(split_point*len(df_learn)),replace=False).index)
        #mask = np.random.rand(len(df_learn)) < split_point    #since by using this method, the length of the df_learn may change, the above method is prefered
        df_train = df_learn[mask]
        df_test = df_learn[~mask]            
        
    #return {'train': df_train, 'test': df_test, 'predict':df_predict} 
    df_train=df_train.reset_index()
    df_test=df_test.reset_index()
    return  df_train, df_test

#%%
def create_predict_dataset(df=None, df_train=None, metadata=None,model=None):
    '''
    creates the predict dataframe based on the information provided in the config file.
    This function may cretes the prediction df using the provided datafrane, a seperate csv file, to try to collects data from external resources. It also runs the spatial filtering for choosing the required segments, polygon, or counties.

    Parameters
    ----------
    df : dataframe
        provided dataframe.
    df_train : dataframe
        the dataframe used for training. It might be used for predicting traffic data
    metadata : dictionary
        metadata
    model : class model
        This is the model trained and can be used. It will be used if the provided df does not include clustering.

    Returns
    -------
    df_predict : dataframe
        prediction dataframe.

    '''
    np.random.seed(metadata['seed_number']) 

    if metadata['predict_data_from_csv_tag']==True:
        print('Predict dataset is collected from the provided csv file. ')
        df_predict=pd.read_csv(metadata['future_data_address'],parse_dates=['time_local'], na_values  = ["?"])
        feature_checker(metadata,df_predict)
    
    else:
        Predicton_Type_Tag=Time_checker(pd.Timestamp(metadata['start_time_predict']).floor(freq='{}H'.format(int(metadata['window_size']/3600)))) 
        #%%
        #Time_checker(pd.Timestamp(year=2020, month=1, day=1, hour=1).floor(freq='4H')) 
        if Predicton_Type_Tag== 'Historical':
            if (metadata['train_test_type']=='simple') | (metadata['train_test_type']=='moving_window_month') | (metadata['train_test_type']=='moving_window_week'):
                #print('The predict dataset is defined just based on the input dates.')
                df_predict = df.loc[(df[metadata['time_column']] >= metadata['start_time_predict'])    &   (df[metadata['time_column']] < metadata['end_time_predict']) ]
                
            elif metadata['train_test_type']=='random_pseudo':
                df_predict = df.loc[(df[metadata['time_column']] >= metadata['start_time_predict'])    &   (df[metadata['time_column']] < metadata['end_time_predict']) ]
            
            elif metadata['train_test_type']=='ratio_random':               
                df_predict = df.loc[(df[metadata['time_column']] >= metadata['start_time_predict'])    &   (df[metadata['time_column']] < metadata['end_time_predict']) ]
                
            #spatial filtering    
            df_predict,metadata['segment_list_pred'] =Spatial_Filtering(df_predict,metadata)
                
            #Adding Cluster if needed
            #if model is defined then we know the clustering type so we can add the cluster_label here. Otherwise we just skip. 
            directory=metadata['Input_Address']+"/trained_models/clusters"+"/"+model.metadata['Cluster_Name']
            Cluster = pickle.load(open(directory+'.sav', 'rb'))
            df_predict= Cluster.pred(df_predict,metadata)                  
            
        #%%
        elif Predicton_Type_Tag== 'Future':
            if df_train is None:
                try:
                    directory=metadata['Input_Address']+"/resampled_clustered_traintestsplit/"+model.metadata['Data_Name']
                    df_train=pd.read_pickle(directory+'/train.pkl')
                except:
                    directory=metadata['Input_Address']+"/resampled_clustered_traintestsplit/"+metadata['current_model']['Data_Name']
                    df_train=pd.read_pickle(directory+'/train.pkl')
            metadata['features_ALL'] = (metadata['features']).copy()
            metadata['features_ALL'] = update_meta_features(metadata['features_ALL'], df_features=df_train.columns.tolist(), cat_col=metadata['cat_features'])

            #spatial filtering 
            df_train,metadata['segment_list_pred'] =Spatial_Filtering(df_train,metadata)
            df_predict=get_future_data(metadata,df_train)


            #metadata['cat_features']=['window','Weekend_or_not']

            #adding categorical data. This step requires the train dataset
            All_col=df_train.columns.tolist()
            for cat_col in metadata['cat_features']:
                cat_converted_cols=[i for i in All_col if 'cat_'+cat_col in i]
                #print(df_train[[cat_col]+cat_converted_cols].columns)
                df_train_nodop=df_train[[cat_col]+cat_converted_cols].drop_duplicates()
                df_predict=pd.merge(df_predict,df_train_nodop, left_on=cat_col, right_on=cat_col, how='left' )

            
            for feature in metadata['features_ALL']:
                if feature in df_predict.columns:
                    pass
                else:
                    print('The model was not able to collect {}. It is considered zero.')
                    df_predict[feature]=0
            #Adding Cluster if needed
            #if model is defined then we know the clustering type so we can add the cluster_label here. Otherwise we just skip. 
            '''
            try: 
                directory=metadata['Output_Address']+"/trained_models/clusters"+"/"+model.metadata['Cluster_Name']                    
                Cluster = pickle.load(open(directory+'.sav', 'rb'))
                df_predict= Cluster.pred(df_predict,metadata)                  
            except:
                pass   
            '''                    



    df_predict=df_predict.reset_index().drop('index',axis=1)
    '''
    if not metadata['pred_name_Count'] in df_predict.columns:
        df_predict[metadata['pred_name_Count']]=None
    if not metadata['pred_name_TF'] in df_predict.columns:
        df_predict[metadata['pred_name_TF']]=None    
    '''
    
    if len(df_predict)==0:
        raise ValueError("the prediction df is empty. Please broaden time/space range.")  
    
    return df_predict
    
#%%    
def Train_Verification_Split(df_learn,metadata):
    '''
    This function splits the df_learn into train and validation datasets.

    Parameters
    ----------
    df_learn : dataframe
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    df_train : dataframe
        The dataframe can be used for training.
    df_verif : dataframe
        The dataframe can be used for validation.

    '''
    
   
    #For now I just include a simple random method, but we can use other methods like cross validation later if it is needed
    split_point = metadata['train_verification_split']
    df_train=pd.DataFrame()
    df_verif=pd.DataFrame()
    clusters = sorted(df_learn.cluster_label.unique())
    for temp_cluster in clusters:
        np.random.seed(metadata['seed_number']) 
        #df_learn_cluster= df_learn    [df_learn['cluster_label']==temp_cluster].reset_index().drop('index',axis=1)
        df_learn_cluster = df_learn.loc[df_learn.cluster_label   == temp_cluster]
        mask=df_learn_cluster.index.isin(df_learn_cluster.sample(int(split_point*len(df_learn_cluster)),replace=False).index)        
        df_train = df_train.append(df_learn_cluster[mask] , ignore_index=True)
        df_verif = df_verif.append(df_learn_cluster[~mask], ignore_index=True)
        
        
    df_train=df_train.sort_values('ID_')
    df_verif=df_verif.sort_values('ID_')
    return df_train,df_verif    
