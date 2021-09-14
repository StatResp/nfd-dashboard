"""
@Author - Sayyed Mohsen Vazirizade  s.m.vazirizade@vanderbilt.edu
This file includes all the functions can be used to check the validity of the input configuration. 
"""
from datetime import datetime
import datetime as dt
import pandas as pd
from os import path



def Input_checker(metadata):
    '''
    This function checks that the values using the other checker functions to verify the parameteres defined in the metadata.

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    #1) checking the parameters in the metadata
    Parameter_Checker(metadata)
    #2) checking the input files
    File_Checker(metadata)
    #3) checking the spatial dimensions
    Space_Checker(metadata)
    #4) checking the temporal dimensions
    #For train
    for i in [metadata['start_time_train_abs'],metadata['end_time_train_abs']]:
        Time_checker(i)
    #For test
    if metadata['train_test_type'] in ['simple','moving_window_month', 'moving_window_week']:
        for i in [metadata['start_time_test_abs'],metadata['end_time_test_abs']]:
            Time_checker(i)  
    #For predict
    if metadata['predict_data_from_csv_tag']==False:
        for i in  [metadata['start_time_predict'],metadata['end_time_predict']]:
            Time_checker(i)      

def Space_Checker(metadata):
    '''
    This function checks that the values (regarding spacial dimension) in the metadata are correct.

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    Dict= {'knox': 2,
    'washington': 8,
    'carter': 47,
    'jefferson': 25,
    'claiborne': 61,
    'loudon': 27,
    'greene': 19,
    'sevier': 28,
    'roane': 29,
    'blount': 13,
    'sullivan': 6,
    'hawkins': 34,
    'monroe': 36,
    'anderson': 18,
    'hamblen': 41,
    'grainger': 68,
    'campbell': 49,
    'union': 80,
    'scott': 87,
    'hancock': 73,
    'morgan': 77,
    'cumberland': 23,
    'cocke': 37,
    'johnson': 82,
    'hamilton': 3,
    'rhea': 59,
    'bradley': 14,
    'unicoi': 71,
    'mcminn': 24,
    'polk': 56,
    'meigs': 84,
    'obion': 20,
    'weakley': 22,
    'lake': 93,
    'marion': 15,
    'lincoln': 33,
    'lawrence': 50,
    'sumner': 7,
    'smith': 51,
    'williamson': 4,
    'davidson': 1,
    'rutherford': 5,
    'lewis': 79,
    'putnam': 16,
    'wilson': 12,
    'sequatchie': 70,
    'bledsoe': 89,
    'maury': 11,
    'giles': 26,
    'dickson': 21,
    'bedford': 53,
    'hickman': 42,
    'montgomery': 10,
    'franklin': 48,
    'marshall': 39,
    'overton': 55,
    'robertson': 17,
    'cannon': 83,
    'grundy': 57,
    'warren': 40,
    'perry': 81,
    'houston': 90,
    'jackson': 85,
    'van buren': 86,
    'macon': 75,
    'cheatham': 66,
    'humphreys': 64,
    'dekalb': 74,
    'stewart': 60,
    'clay': 88,
    'coffee': 43,
    'wayne': 52,
    'fentress': 65,
    'shelby': 0,
    'decatur': 69,
    'chester': 76,
    'madison': 9,
    'dyer': 31,
    'gibson': 30,
    'hardeman': 46,
    'henry': 44,
    'pickett': 94,
    'carroll': 38,
    'hardin': 54,
    'mcnairy': 45,
    'henderson': 58,
    'crockett': 63,
    'benton': 72,
    'lauderdale': 67,
    'tipton': 62,
    'haywood': 35,
    'fayette': 32,
    'white': 78,
    'moore': 91,
    'trousdale': 92}
    
    for i in metadata['segment_list_pred']:
        if isinstance(i, (int, float)):
            pass
        elif i=='All' or i=='all':
            pass
        elif not i in Dict.keys():
            raise ValueError("{} is not a county.".format(i)) 
        
    if len(metadata['polygon'])==0:
        pass
    elif len(metadata['polygon'])!=4:
        raise ValueError("polygon should be a rectangle defined by 4 points")
    elif len(metadata['polygon'])==4:
        for i in metadata['polygon']:
            if len(i)!=2:
                raise ValueError("polygon is not acceptable. Each point should be pair coordinates.")
            if i[0]>0 or i[1]<0:
                raise ValueError("The Lon and Lat is not acceptable for the polygon. They might be flipped.")


def Parameter_Checker(metadata):
    '''
    This function checks that the values in the metadata are correct.

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for i in metadata['model_type']:
        if not i in ['LR','RF','NN', 'ZIP']:
            raise ValueError("{} is not acceptable.".format('model_type')) 
    
    for i in metadata['cluster_type']:
        if not i in ['NoC','KM','AH']:
            raise ValueError("{} is not acceptable.".format('cluster_type'))

    for i in metadata['cluster_number']:
        if (not isinstance(i, int)) and (i!=''):
            print(metadata['cluster_number'])
            raise ValueError("{} is not acceptable.".format('cluster_number'))

    for i in metadata['resampling_type']:
        if not i in ['NR','ROS','RUS']:
            raise ValueError("{} is not acceptable.".format('resampling_type'))

    for i in metadata['resampling_rate']:
        if not ((i>=0 and i<=1) or (i==-1)):
            raise ValueError("{} is not acceptable.".format('resampling_rate'))
            
    if not metadata['train_test_type'] in ['simple', 'ratio_random', 'random_pseudo', 'moving_window_month', 'moving_window_week']:
        raise ValueError("{} is not acceptable.".format('train_test_type'))

    Total_len=len(metadata['model_type'])

    if  (len(metadata['model_hyperparam_address']) != Total_len) or (len(metadata['cluster_type']) != Total_len) or (len(metadata['cluster_number']) != Total_len) or (len(metadata['resampling_type']) != Total_len) or (len(metadata['resampling_rate']) != Total_len):
        raise ValueError("{} is not acceptable.".format('length of the lists for model_type, hyperparam_config_files, cluster_type, cluster_number, resampling_type, and  resampling_rate should be the same.'))
           
        
def File_Checker(metadata):
    '''
    This function checks if all the required files exist in the intented addresses.

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if not path.exists(metadata['merged_pickle_address']):
        raise ValueError("file, {}, does not exist.".format(metadata['merged_pickle_address'])) 

    if metadata['figure_tag']==True:
        if not path.exists(metadata['inrix_pickle_address']):
            raise ValueError("file, {}, does not exist, and it is required for plotting.".format(metadata['inrix_pickle_address'])) 
           
            
def Time_checker(Beg):
    '''
    This function checks the time range of the input data

    Parameters
    ----------
    Beg : 
        beggining time.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    Predicton_Type_Tag : TYPE
        DESCRIPTION.

    '''
    Earliest_Available_Time=pd.Timestamp(year=2017, month=5, day=1, hour=0)
    Latest_Available_Time  =pd.Timestamp(year=2021, month=6, day=1, hour=0)+pd.DateOffset(seconds=1)
    
    if (Beg< Earliest_Available_Time):
        print('The requested date, {}, is in not in the acceptable range.'.format(Beg))
        print('For historical prediction acceptable range is between {} and {}'.format(Earliest_Available_Time,Latest_Available_Time))
        raise ValueError("The date range is too past")
    elif (Beg>= Earliest_Available_Time) and (Beg< Latest_Available_Time) :
        print('The requested date, {}, is in the acceptable range for Historical prediction.'.format(Beg))
        Predicton_Type_Tag='Historical'           
    elif (Beg>= Latest_Available_Time) and (Beg< pd.Timestamp.now().floor(freq='4H')) :
        print('The requested date, {}, is in not in the acceptable range.'.format(Beg))
        print('For historical prediction acceptable range is between {} and {}'.format(Earliest_Available_Time,Latest_Available_Time))
        raise ValueError("The date range is not acceptable. Date after Aug/24/2020 and before now is not acceptable.")
    elif (Beg>= pd.Timestamp.now().floor(freq='4H')) and (Beg< pd.Timestamp.now().floor(freq='4H')+pd.DateOffset(days=5)) :
        print('The requested date, {}, is in the acceptable range for future prediction.'.format(Beg))
        Predicton_Type_Tag='Future'
    elif (Beg>= pd.Timestamp.now().floor(freq='4H')+pd.DateOffset(days=5)) :
        print('The requested date, {}, is in not in the acceptable range. The model can predict upto 5 days'.format(Beg))
        print('For future prediction acceptable range is between {} and {}'.format(str(dt.date.today().strftime('%b/%d/%Y')),str((dt.date.today()+dt.timedelta(days=5)).strftime('%b/%d/%Y'))))
        raise ValueError("The date range is too future.")
    return Predicton_Type_Tag            



def feature_checker(metadata, df):
    '''
    This function checks for the existence of the features on the df 

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    if not 'time_local' in df.columns:
            raise ValueError("feature {} does not exist!".format('time_local'))    
    
    if not metadata['unit_name'] in df.columns:
            raise ValueError("feature {} does not exist!".format(metadata['unit_name']))


    if not metadata['features_temporal'] is None:
        for i in metadata['features_temporal']:
            if (not i in df.columns):
                print(i)
                raise ValueError("feature {} does not exist!".format(i))
    if not metadata['features_incident'] is None:
        for i in metadata['features_incident']:
            if (not i in df.columns):
                print(i)
                raise ValueError("feature {} does not exist!".format(i))
    if not metadata['features_weather'] is None:
        for i in metadata['features_weather']:
            if (not i in df.columns):
                print(i)
                raise ValueError("feature {} does not exist!".format(i))
    if not metadata['features_traffic'] is None:
        for i in metadata['features_traffic']:
            if (not i in df.columns):
                print(i)
                raise ValueError("feature {} does not exist!".format(i))
    if not metadata['features_static'] is None:
        for i in metadata['features_static']:
            if (not i in df.columns):
                print(i)
                raise ValueError("feature {} does not exist!".format(i))