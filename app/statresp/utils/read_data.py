# -*- coding: utf-8 -*-
"""
@Author - Sayyed Mohsen Vazirizade  s.m.vazirizade@vanderbilt.edu
This file includes all the functions can be used to import datasets. 
"""

from statresp.ml.forecasters.utils import dummy_maker
from statresp.utils.checkers import feature_checker
import pandas as pd
import os
import csv



def prepare_merged_dataset(metadata):
    #for reading the incident data frame from a pickle file 
    print('reading ...')
    if metadata['merged_pickle_address'].split('.')[-1]=='pkl':
        df_merged= pd.read_pickle(open(metadata['merged_pickle_address'], 'rb'))
    else:
        df_merged= pd.read_parquet(metadata['merged_pickle_address'])

    #df_grouped=pd.read_pickle(metadata['inrix_pickle_address'])
    
    
    #we reset the index since in man cases it is not unique
    df_merged=df_merged.reset_index().rename(columns={'index':'ID_'})
    #deleting some unnecessary  columns may exist in the data
    if 'time' in df_merged.columns:
                df_merged=df_merged.drop('time',axis=1) 
                
    if 'nearest_weather_station' in     df_merged.columns:
        df_merged=df_merged.rename(columns={'nearest_weather_station':'Nearest_Weather_Station'})         
    if 'timelocal' in     df_merged.columns:
        df_merged=df_merged.rename(columns={'timelocal':'time_local'})     
    if 'grouped_type3' in     df_merged.columns:
        df_merged=df_merged.rename(columns={'grouped_type3':metadata['unit_name']})     

    if not 'Nearest_Weather_Station' in df_merged.columns:
        print('Nearest_Weather_Station does not exist in {} so the code looks for it in {}'.format(metadata['merged_pickle_address'],metadata['inrix_pickle_address']))

        if metadata['inrix_pickle_address'][-4:] == '.pkl':
            df_inrix = pd.read_pickle(metadata['inrix_pickle_address'])
        else:
            df_inrix = pd.read_parquet(metadata['inrix_pickle_address'])
            uncompatible_list = ['beg', 'end', 'center', 'geometry', 'geometry3d', 'geometry_highres']
            for i in uncompatible_list:
                if i in df_inrix.columns:
                    # df[i] = df[i].astype(str)
                    from shapely import wkt
                    df_inrix[i] = df_inrix[i].apply(wkt.loads)

        df_inrix=df_inrix[[metadata['unit_name'],'Nearest_Weather_Station']].drop_duplicates(metadata['unit_name'])
        df_merged=pd.merge(df_merged,df_inrix, left_on=metadata['unit_name'], right_on=metadata['unit_name'],how='left' )

    
    # some of the modules raise errors with bool type. We convert them here. 
    for col in df_merged.columns:
        #print(col)
        if df_merged.dtypes[col]==bool:
            df_merged[col]=df_merged[col].astype(int)


    if 'index' in     df_merged.columns:
        df_merged=df_merged.drop('index',axis=1)    
    
    feature_checker(metadata,df_merged)
    
    
    mandatory_features=['ID_',metadata['unit_name'], 'time_local', 'Nearest_Weather_Station', 'county', metadata['pred_name_TF'], metadata['pred_name_Count'] ]
    All_features=mandatory_features
    if not metadata['features_temporal'] is None:
        All_features = All_features+metadata['features_temporal']
    if not metadata['features_incident'] is None:
        All_features = All_features+metadata['features_incident']
    if not metadata['features_weather'] is None:
        All_features = All_features+metadata['features_weather']
    if not metadata['features_traffic'] is None:
        All_features = All_features+metadata['features_traffic']
    if not metadata['features_static'] is None:
        All_features = All_features+metadata['features_static']
    df_merged=df_merged[All_features]
    
    df_merged=dummy_maker(df_merged,metadata['cat_features'])
    
    return df_merged



def Read_data(Name,directory,Type=None): 
    '''
    The datasets created by run_dataprep can be called using this function. 

    Parameters
    ----------
    Name : string
        The name of the type of dataset. For example, ROS0.5+AH7-2017-05-01-000000to2020-01-07-000000
    directory : TYPE
        DESCRIPTION.
    Type : string
        If you want to read a specific file, you can define here.

    Returns
    -------
    df_train : dataframe
        DESCRIPTION.
    df_verif : dataframe
        DESCRIPTION.
    df_test : dataframe
        DESCRIPTION.
    df_predict : dataframe
        DESCRIPTION.

    '''
    
    
    #Name=metadata['current_model']['Data_Name']
    

    #directory="output/resampled_clustered_traintestsplit/"+Name
    if Type is None:
        df_train=pd.read_pickle(directory+Name+'/train.pkl')
        df_verif=pd.read_pickle(directory+Name+'/validation.pkl')
        df_test=pd.read_pickle(directory+Name+'/test.pkl')
        df_predict=pd.read_pickle(directory+Name+'/predict.pkl')
        return df_train, df_verif, df_test, df_predict
    else:
        if Type=='train':
            df_train=pd.read_pickle(directory+Name+'/train.pkl')
            return df_train
        elif Type=='validation':
            df_verif=pd.read_pickle(directory+Name+'/validation.pkl')
            return df_verif
        elif Type=='test':
            df_test=pd.read_pickle(directory+Name+'/test.pkl')
            return df_test
        elif Type=='predict':       
            df_predict=pd.read_pickle(directory+Name+'/predict.pkl')
            return df_predict



def Savedata(df_train=None, df_verif=None,df_test=None, df_predict=None, Name=None,directory=None): 
    '''
    The datasets created by run_dataprep can be saved using this function. 

    Parameters
    ----------
    Name : string
        The name of the type of dataset. For example, ROS0.5+AH7-2017-05-01-000000to2020-01-07-000000
    directory : TYPE
        DESCRIPTION.

    Returns
    -------
    df_train : dataframe
        DESCRIPTION.
    df_verif : dataframe
        DESCRIPTION.
    df_test : dataframe
        DESCRIPTION.
    df_predict : dataframe
        DESCRIPTION.

    '''
    
    #directory=metadata['Output_Address']+"/resampled_clustered_traintestsplit/"+
    directory=directory+"/resampled_clustered_traintestsplit"+'/'+Name
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not df_train is None:
        df_train.to_pickle(directory+'/train.pkl')
        # print('train.pkl is saved.')
    if not df_verif is None:
        df_verif.to_pickle(directory+'/validation.pkl')
        # print('validation.pkl is saved.')
    if not df_test is None:
        df_test.to_pickle(directory+'/test.pkl')
        # print('test.pkl is saved.')
    if not df_predict is None:
        df_predict.to_pickle(directory+'/predict.pkl')
        # print('predict.pkl is saved.')



def check_resampled_clustered_traintestsplit_data_exist(Name,directory):
    '''
    It checks if all of the datasets exist or not.

    Parameters
    ----------
    Name : string
        The name of the type of dataset. For example, ROS0.5+AH7-2017-05-01-000000to2020-01-07-000000
    directory : string
        The location the user expect the files are located.

    Returns
    -------
    df_train : dataframe
        DESCRIPTION.
    df_verif : dataframe
        DESCRIPTION.
    df_test : dataframe
        DESCRIPTION.
    df_predict : dataframe
        DESCRIPTION.

    '''
    directory=directory+"/resampled_clustered_traintestsplit"+'/'+Name
    if os.path.exists(directory+'/train.pkl') and os.path.exists(directory+'/validation.pkl') and os.path.exists(directory+'/test.pkl'):
        return True
    return False


def check_trainmodel_exist(Name, directory):
    '''
    It checks if the trained model exists or not.

    Parameters
    ----------
    Name : string
        The name of the type of dataset. For example, LR+AH7-2017-05-01-000000to2020-01-07-000000.sav
    directory : string
        The location the user expect the files are located.

    Returns
    -------
    True or False.
    '''

    directory = directory + "/trained_models/models" + '/' + Name + '.sav'
    if os.path.exists(directory):
        return True
    return False


def Save_training_results(DF_Test_metric_time,DF_performance,DF_Test_spacetime, model_set,Address,Format):
    '''
    This function saves the results generated by the run_ml.py

    Parameters
    ----------
    
    DF_Test_metric_time : dataframe
        This dataframe includes the performance metrics for each time window (and all segments) and for each model. The main metrics are accuracy, recall, precision, F1 score, and Correctness. In this dataframe, the rows are time windows and columns are performance metrics of each model. For example, if we 2 modes (LR+AH7 and NN+KM2), 100 segments, 4-h time resolution, the rows of this dataframe are the 4-h time windows, and the columns are accuracy, recall, precision, F1 score, and Correctness for each model. 
        
    DF_performance  : dataframe
        The total performance metrics for each model are stored in this dataframe. If the rolling window option for training and testing is selected, it includes the performance for each rolling window separately, and the average of all of them is appeneded at the end. It includes the log-likelihood, total accuracy, precision, recall, and F1 score, Pearson and spearman correlation, MSE, Correctness, etc. The rows are (rolling) test windows and models and the columns are metrics.  
        
    DF_Test_spacetime : dataframe
        It includes all of the predicted values for the test dataset. The rows are time windows and roadway segments. The columns are the values predicted by each model. If the rolling window option for training and testing is selected, the prediction values of different rolling windows are appended to each other. 
        
    DF_results_Mean-All-Window.xlsx: excel file
        For the convenience of the user, the average performance of all models over all test windows is summarized here. In other words, this is the same as the average of performance metrics for each model available in DF_Test_metric_time.  
        
    model_set : list
        a list of the name of the trained models.     
    
    metadata : dictionary
        metadata.

    Returns
    -------
    None.

    '''


    #Address=metadata['Output_Address']+'/train'
    if not os.path.exists(Address):
             os.makedirs(Address)           
    print("Saving Training Results... \n \n")
    
    
    if Format=='pkl':
        DF_performance.to_pickle(Address+'/DF_performance'+'.pkl')  
        DF_Test_spacetime.to_pickle(Address+'/DF_likelihood_spacetime'+'.pkl')
        DF_Test_metric_time.to_pickle(Address+'/DF_performance_time'+'.pkl')
    
    elif Format=='csv':
        DF_performance.to_csv(Address+'/DF_performance'+'.csv')  
        DF_Test_spacetime.to_csv(Address+'/DF_likelihood_spacetime'+'.csv')
        DF_Test_metric_time.to_csv(Address+'/DF_performance_time'+'.csv')
    
    elif Format=='json':
        DF_performance.to_json(Address+'/DF_performance'+'.csv', orient='table')
        DF_Test_spacetime.to_json(Address+'/DF_likelihood_spacetime'+'.csv', orient='table')
        DF_Test_metric_time.to_json(Address+'/DF_performance_time'+'.csv', orient='table')

    elif Format=='parquet':
        DF_performance.to_parquet(Address+'/DF_performance'+'.csv')
        DF_Test_spacetime.to_parquet(Address+'//DF_likelihood_spacetime'+'.csv')
        DF_Test_metric_time.to_parquet(Address+'/DF_performance_time'+'.csv')


    with pd.ExcelWriter(Address+'/DF_results' + '_Mean-All-Window.xlsx') as writer:
        for i in set(DF_performance['rolling_window_group'].tolist()):
            DF_performance[DF_performance['rolling_window_group'] ==i][
                ['rolling_window_group', 'model', 'train_likelihood', 'test_likelihood', 'accuracy', 'precision',
                 'recall', 'f1', 'spearman_corr', 'pearson_corr', 'Correctness']].to_excel( writer, sheet_name=i, index=False)


    #DF_performance[DF_performance['rolling_window_group']=='Mean'][['rolling_window_group', 'model', 'train_likelihood', 'test_likelihood', 'accuracy', 'precision', 'recall', 'f1', 'spearman_corr', 'pearson_corr', 'Correctness']].to_excel(Address+'/DF_results' + '_Mean-All-Window.xlsx',index=False)
    
    with open(Address+'/model_names.csv', 'w') as f:
       write = csv.writer(f)
       write.writerow(model_set) 

    print('Done Saving')


                
def read_model_list(metadata):
        with open(metadata['Input_Address']+'/'+metadata['simulation']['source']+'/model_names.csv', 'r') as f:
            read = csv.reader(f)
            model_list = []
            for row in read:
                #print(row)
                model_list.extend(row)
        return model_list    
    
    
    
    
    
    
    
    
    
    
    