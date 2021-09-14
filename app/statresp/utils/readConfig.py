'''
@Author - Sayyed Mohsen Vazirizade s.m.vazirizade@vanderbilt.edu
Parent file for all the functions required to read the config file or the user input from windows/macos/linux host
'''
from statresp.utils.checkers import Input_checker
from statresp.utils.settings import ArgParser,print_l,Singleton
import configparser
from datetime import datetime
import datetime as dt
import pytz
from pprint import pprint
import pandas as pd
import numpy as np
import json
from multiprocessing import Pool, cpu_count
from os import path


def ConfigSectionMap(config, section):
    """
    Reads the params.conf file into a dictionary
    @param config: parsed config
    @param section: section in the config file
    @return:
    """
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                print(("skip: %s" % option))
        except:
            print(("exception on %s!" % option))
            dict1[option] = None
    return dict1


def FixingTimeZone(metadata):
    '''
        The code works based on local time zone. If The user defined any other time zone in the input, this function converts that to the local time zone. 

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    metadata : TYPE
        DESCRIPTION.

    '''
    if metadata['time_zone'].lower()=='local':
        print('Time is set to be local.')
    else:
        pst=pytz.timezone(metadata['time_zone'])
        for TimeLabel in ['start_time_train_abs','end_time_train_abs','start_time_test_abs','end_time_test_abs', 'start_time_predict', 'end_time_predict']:
            metadata[TimeLabel]=pst.localize(metadata[TimeLabel].replace(tzinfo=None))
    return metadata





def create_metadata(config):
    """
    Takes the parsed config file and creates metadata
    @param config: parsed config
    @return: metadata in dictionary form
    """
    metadata = {
        'incident_pickle_address': ConfigSectionMap(config, "filepaths")["incident_pickle"],
        #'traffic_pickle_address': ConfigSectionMap(config, "filepaths")["traffic_pickle"],
        #'weather_pickle_address': ConfigSectionMap(config, "filepaths")["weather_pickle"],
        'inrix_pickle_address': ConfigSectionMap(config, "filepaths")["inrix_pickle"],
        #'groups_pickle_address': ConfigSectionMap(config, "filepaths")["groups_pickle"],
        'merged_pickle_address': ConfigSectionMap(config, "filepaths")["merged_pickle"],
        'graph_pickle_address': ConfigSectionMap(config, "filepaths_advanced")["graph_pickle"],
        'future_data_address': ConfigSectionMap(config, "prediction_advanced")["prediction_dataset_csv_address"],
        'window_size': int(ConfigSectionMap(config, "metadata")["window_size"]),
        'unit_name': ConfigSectionMap(config, "metadata")["unit_name"],
        'pred_name_Count': ConfigSectionMap(config, "metadata")["pred_name_count"],
        'pred_name_TF': ConfigSectionMap(config, "metadata")["pred_name_tf"],
        'time_zone': ConfigSectionMap(config, "metadata")["time_zone"],
        'figure_tag':(ConfigSectionMap(config, "metadata")["figure_tag"]=='True'),
        'seed_number': int(ConfigSectionMap(config, "metadata")["seed_number"]),
        'train_test_type': str(ConfigSectionMap(config, "metadata")["train_test_type"]),
        'train_test_split': float(ConfigSectionMap(config, "metadata")["train_test_split"]),
        'train_verification_split': float(ConfigSectionMap(config, "metadata")["train_verification_split"]),
        'start_time_train_abs': datetime.strptime(ConfigSectionMap(config, "metadata")["start_time_train"], '%Y-%m-%d %H:%M:%S'),
        'end_time_train_abs': datetime.strptime(ConfigSectionMap(config, "metadata")["end_time_train"],     '%Y-%m-%d %H:%M:%S'),
        'start_time_test_abs': datetime.strptime(ConfigSectionMap(config, "metadata")["start_time_test"], '%Y-%m-%d %H:%M:%S'),
        'end_time_test_abs': datetime.strptime(ConfigSectionMap(config, "metadata")["end_time_test"],     '%Y-%m-%d %H:%M:%S'),
        'start_time_predict': datetime.strptime(ConfigSectionMap(config, "prediction")["start_time_predict"],   '%Y-%m-%d %H:%M:%S'),
        'end_time_predict': datetime.strptime(ConfigSectionMap(config, "prediction")["end_time_predict"],       '%Y-%m-%d %H:%M:%S'),
        'predict_data_from_csv_tag':(ConfigSectionMap(config, "prediction_advanced")["predict_data_from_csv"]=='True'),
        'naive': (ConfigSectionMap(config, "mlModels")["naive"]=='True'),
        'model_to_predict': ConfigSectionMap(config, "prediction")["model_to_predict"],
        'weatherbit_key': ConfigSectionMap(config, "keys")["weatherbit_key"],
    }
    metadata=FixingTimeZone(metadata)
    
    
    #Regeression Parameters
    #1) Parameters from DFs
    metadata['features']=[]
    f       = ConfigSectionMap(config, "features")
    for Name in ['features_temporal', 'features_incident', 'features_weather', 'features_traffic','features_static']:
        try: 
            metadata[Name] = [x.strip() for x in f[Name].split(',')]
            metadata['features'].extend(metadata[Name])
        except:   
            metadata[Name] = None
    #3) Categorical Params
    try:    
        metadata['cat_features'] = [x.strip() for x in f["categorical_features"]  .split(',')]
    except:
        metadata['cat_features']=None        
    #4) train test split for the training of reg model
    #metadata['train_test_split']    = float(f["train_test_split"])



    #f_stat  = ConfigSectionMap(config, "clusterParams")["static_features"]
    #metadata['Cluster_type']  = ConfigSectionMap(config, "clusterParams")["cluster_type"]
    #metadata['Cluster_numbers']  = int(ConfigSectionMap(config, "clusterParams")["cluster_numbers"])

    f_clusters= ConfigSectionMap(config, "clusterParams")
    metadata['cluster_type']  = [x.strip() for x in f_clusters['cluster_type'].split(',')]  
    metadata['cluster_number']  = [int(x) for x in f_clusters['cluster_number'].split(',')]  
    metadata['cluster_number'] = ['' if i==-1  else i for i in metadata['cluster_number'] ]

    f_resamples= ConfigSectionMap(config, "resamplingParams")
    metadata['resampling_type']  = [x.strip() for x in f_resamples['resampling_type'].split(',')]  
    metadata['resampling_rate']  = [float(x)  for x in f_resamples['resampling_rate'].split(',')]  
    metadata['resampling_rate'] = ['' if i==-1  else i for i in metadata['resampling_rate'] ]


    f_models= ConfigSectionMap(config, "mlModels")["model_type"]
    metadata['model_type'] = [x.strip() for x in f_models.split(',')]
     
    
    
    return metadata


def read_config(Config_Address='config/config.conf',InputFolder_Address='data',OutputFolder_Address='output'):
    """
    reads the config file and returns metadata dictionary
    @return:
    """
    # READ CONFIG
    config = configparser.ConfigParser()
    config.read(Config_Address)
    metadata = create_metadata(config)
    
    metadata['incident_pickle_address'] = InputFolder_Address+'/'+metadata['incident_pickle_address']
    metadata['inrix_pickle_address'] = InputFolder_Address+'/'+metadata['inrix_pickle_address']
    metadata['merged_pickle_address'] = InputFolder_Address+'/'+metadata['merged_pickle_address']
    metadata['future_data_address'] = InputFolder_Address+'/'+metadata['future_data_address']
    metadata['graph_pickle_address'] = InputFolder_Address + '/' + metadata['graph_pickle_address']

    metadata['Input_Address']=InputFolder_Address
    metadata['Config_Address']=Config_Address
    metadata['Output_Address']=OutputFolder_Address


    model_params_address = json.loads(config.get("mlModels","hyperparam_config_files"))
    metadata['model_hyperparam_address'] = model_params_address
    segment_list = json.loads(config.get("prediction", "segment_list"))
    metadata['segment_list_pred'] = segment_list
    county_list = json.loads(config.get("prediction", "county_list"))
    metadata['county_list_pred'] = county_list
    polygon = json.loads(config.get("prediction","polygon"))
    metadata['polygon'] = polygon
    if (metadata['time_zone']).lower()=='local':
        metadata['time_column']='time_local'
    elif (metadata['time_zone']).lower()=='utc':
        metadata['time_column']='time'    
    
    #metadata['units'] = np.sort(df_[metadata['unit_name']].unique()) #finding the unique metadata['unit_name'], which means unique locations
    #metadata['location_map'] = {x: 1 for x in metadata['units']}    
    
    metadata['simulation']=dict()
    metadata['simulation']['source'] = ConfigSectionMap(config, "simulation")["source"]
    metadata['simulation']['num_ambulances'] = json.loads(config.get("simulation","num_ambulances"))
    metadata['simulation']['alpha'] = json.loads(config.get("simulation","alpha"))
    metadata['simulation']['average_service_time'] = json.loads(config.get("simulation","average_service_time"))
    metadata['simulation']['ambulance_avg_speed'] = json.loads(config.get("simulation","ambulance_avg_speed"))
    metadata['simulation']['penalty'] = json.loads(config.get("simulation","penalty"))
    metadata['simulation']['num_cores']= int(ConfigSectionMap(config, "simulation")["num_cores"])
    if metadata['simulation']['num_cores']==0:
        metadata['simulation']['num_cores']=cpu_count()-1
    metadata['simulation']['grid_size'] = float(ConfigSectionMap(config, "simulation_advanced")["grid_size"])
    Input_checker(metadata)
          
    
    return metadata


def add_to_metadata_modeltype(metadata):
    metadata['TF_Models']=['LR','RF','NN','SVM']
    metadata['Count_Models']=['ZIP','SR','PR']
    metadata['GLM_Models']=['SR','LR','PR', 'NBR','ZIP']
    
    metadata['available_resampling_methods']=['ROS', 'RUS', 'SMOTE']
    metadata['available_clustering_methods']=['AH', 'KM']
    return metadata



def set_setting():
    print('Process Initiates:')
    getcwd=os.getcwd()
    os.chdir(getcwd)
    print('getcwd:      ', getcwd)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100)
    random.seed(metadata['seed_number'])
    np.random.seed(metadata['seed_number'])    

def metadata_creator():
    '''
    Use the aforementioned function to read the config file and create a proper metadata from that can be used from all other functions later.

    Returns
    -------
    metadata : TYPE
        DESCRIPTION.

    '''
    
    Input = ArgParser()
    #Config_Address="config/params.conf"
    InputFolder_Address=Input.parse_args().input_folder
    Config_Address=Input.parse_args().config
    OutputFolder_Address=Input.parse_args().output_folder
    rewirte_Tag = Input.parse_args().write
    if rewirte_Tag=='False' or rewirte_Tag=='false' or rewirte_Tag==False:
        rewirte_Tag=False
    elif rewirte_Tag=='True' or rewirte_Tag=='true' or rewirte_Tag==True:
        rewirte_Tag=True
    else:
        raise ValueError("-w is not a correct. Please user either True or False.")
    Singleton.configure_settings(output_folder=OutputFolder_Address)
    metadata = read_config(Config_Address,InputFolder_Address,OutputFolder_Address)
    metadata['rewirte_Tag']=rewirte_Tag
    metadata=add_to_metadata_modeltype(metadata)
    return metadata
    
    

