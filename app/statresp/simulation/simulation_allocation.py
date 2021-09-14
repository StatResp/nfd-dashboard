# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:51:40 2021

@author: Sayyed Mohsen Vazirizade
"""
import time
#import pickle5 as pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#import pygeoj
#import pyproj
from statresp.simulation.allocation.Griding_TN import MyGrouping_Grid, Distance_Dict_Builder
from statresp.simulation.allocation.pmedianAllocator import pmedianAllocator
from statresp.simulation.allocation.Allocation import Dispaching_Scoring, Weight_and_Merge, Responders_Location, Graph_Distance_Metric
from statresp.plotting.figuring_simulation import simulation_animation_Plotly
from multiprocessing import Pool
#import multiprocessing
import json
import shapely.geometry as sg
#import os.path
from os import path
import shapely.wkt
import geopandas as gpd
import os

#%%

def get_allocation(arguments):
    '''
    This function finds the best location for the responders based on solving a modified P-median problem.

    Parameters
    ----------
    arguments : list
        arguments required for the P-median problem .

    Returns
    -------
    None.

    '''


    num_ambulances, alpha, model_i, DF_Test_spacetime, All_seg_incident, Grid_center, df_incident, average_service_time, ambulance_avg_speed, missed_accident_penalty, Window_size_hour, possible_facility_locations, demand_nodes, Distant_Dic, time_range, output_address,unit_name, Figure_Tag = arguments
    #print('num_ambulances: {}, alpha: {}, mlmodel: {}, average_service_time: {}, missed_accident_penalty: {}'.format(num_ambulances, alpha, model_i, average_service_time, missed_accident_penalty))
    df_responders_Exist=False
    #Figure_Tag=metadata['figure_tag']
    #Figure_Tag=False
    All_Responders_GridID={}
    if not os.path.exists(output_address+'/simulation_example/response'):
        os.makedirs(output_address+'/simulation_example/response')
    Address=output_address+'/simulation_example/response/Responder_Location_@'+model_i+'_'+str(num_ambulances)+'V'+str(average_service_time)+'h'+str(ambulance_avg_speed)+'S'+str(alpha)+'alpha.json'
    if path.exists(Address)==True:
        print('{} exist and is not created again'.format(Address))
    else:
        for row,row_values  in time_range.iteritems(): 
                if (model_i!='Naive') | (df_responders_Exist==False) | (len(time_range)>186):
                    print('{} time range: {}, num_ambulances: {}, alpha: {}, mlmodel: {}, average_service_time: {}, missed_accident_penalty: {}'.format(row,row_values,num_ambulances, alpha, model_i, average_service_time, missed_accident_penalty))
                    weights_dict, DF_Test_space_time_i=Weight_and_Merge(DF_Test_spacetime,All_seg_incident,time_i=row_values,unit_name=unit_name,model=model_i)
                    allocator = pmedianAllocator()
                    Responders_GridID = allocator.solve(number_of_resources_to_place=num_ambulances,
                                                 possible_facility_locations= possible_facility_locations,
                                                 demand_nodes=demand_nodes,
                                                 distance_dict=Distant_Dic,
                                                 demand_weights=weights_dict,
                                                 score_type='penalty',
                                                 alpha=alpha)
                    df_responders_Exist=True
                    #print(df_responders[['ID','Grid_ID']])
                else:
                    print(row, row_values, 'for naive model df_responders just generated once')
                #All_Responders_GridID.append(list(Responders_GridID))
                All_Responders_GridID[str(row)]=list(Responders_GridID)
        if not os.path.exists(output_address+'/simulation_example'):
                os.makedirs(output_address+'/simulation_example')
        with open(Address, "w") as f:
                json.dump(All_Responders_GridID, f, indent = 6) 

def get_distance(arguments,Save_Tag=True):
    '''
    This function measures the distance between incidents and responders. 

    Parameters
    ----------
    arguments : list
        arguments required for the P-median problem .

    Returns
    -------
    None.

    '''
    df_incident_responded = pd.DataFrame()
    df_responder_location = pd.DataFrame()
    num_ambulances, alpha, model_i, DF_Test_spacetime, All_seg_incident, Grid_center, df_incident, average_service_time, ambulance_avg_speed, missed_accident_penalty, Window_size_hour, possible_facility_locations, demand_nodes, Distant_Dic, time_range,output_address,unit_name,Figure_Tag = arguments
    #print(num_ambulances, alpha, model_i, average_service_time, missed_accident_penalty, Window_size_hour)
    DF_metric=pd.DataFrame({'time_local':time_range})
    df_responders_Exist=False
    #Figure_Tag=metadata['figure_tag']
    #Figure_Tag=True
    if not os.path.exists(output_address+'/simulation_example/distance'):
        os.makedirs(output_address+'/simulation_example/distance')
    Address=output_address+ '/simulation_example/distance/Distance-Metric_@'  + model_i + '_' + str(num_ambulances) + 'V' + str(average_service_time) + 'h' + str(ambulance_avg_speed) + 'S' + str(alpha) + 'alpha.json'
    if path.exists(Address)==True:
        print('{} exist and is not created again'.format(Address))
        df_incident_responded=pd.read_pickle(output_address + '/simulation_example/distance/incident_responded_@' + model_i + '_' + str(num_ambulances) + 'V' + str(average_service_time) + 'h' + str(ambulance_avg_speed) + 'S' + str(alpha) + 'alpha.pkl')
        df_responder_location=pd.read_pickle(output_address + '/simulation_example/distance/responder_location_@' + model_i + '_' + str(num_ambulances) + 'V' + str(average_service_time) + 'h' + str(ambulance_avg_speed) + 'S' + str(alpha) + 'alpha.pkl')
        Map=simulation_animation_Plotly(df_incident_responded,df_responder_location)
        if Save_Tag == True:
            Map.write_html(output_address + '/simulation_example/incident&responder' +  model_i + '_' + str(num_ambulances) + 'V' + str(average_service_time) + 'h' + str(ambulance_avg_speed) + 'S' + str(alpha) + '_Map_rate.html')
        else:
            return Map
    else:
        with open(output_address+'/simulation_example/response/Responder_Location_@'+model_i+'_'+str(num_ambulances)+'V'+str(average_service_time)+'h'+str(ambulance_avg_speed)+'S'+str(alpha)+'alpha.json') as f:
            All_Responders_GridID = json.load(f)
        for row,row_values  in time_range.iteritems():
            if (model_i!='Naive') | (df_responders_Exist==False)  | (len(time_range)>186):
                print('{} time range: {}, num_ambulances: {}, alpha: {}, mlmodel: {}, average_service_time: {}, missed_accident_penalty: {}'.format(row,row_values,num_ambulances, alpha, model_i, average_service_time, missed_accident_penalty))
                weights_dict, DF_Test_space_time_i=Weight_and_Merge(DF_Test_spacetime,All_seg_incident,time_i=row_values,unit_name=unit_name,model=model_i)
                Responders_GridID=All_Responders_GridID[str(row)]
                df_responders=Responders_Location(Grid_center,Responders_GridID,DF_Test_space_time_i,time_i=row_values,model=model_i, alpha=alpha, Figure_Tag=False)
                df_responders_Exist=True
                #print(df_responders[['ID','Grid_ID']])
            else:
                print(row, row_values, 'for naive model df_responders just generated once')

            Dispatch_Output, df_incident_responded_t, df_responder_location_t=Dispaching_Scoring(row_values,df_responders,df_incident ,Window_size_hour,average_service_time,ambulance_avg_speed,missed_accident_penalty,model_i=model_i,alpha=alpha, Figure_Tag=Figure_Tag)
            df_incident_responded = df_incident_responded.append(df_incident_responded_t, ignore_index = True)
            df_responder_location = df_responder_location.append(df_responder_location_t, ignore_index = True)
            DF_metric.loc[row, list(Dispatch_Output.keys())]=list(Dispatch_Output.values())
            #Figure_Tag=False
        #DF_metric.mean().to_json('results/Distance/Distance-Metric_@'+model_i+'_'+str(num_ambulances)+'V'+str(average_service_time)+'h'+str(ambulance_avg_speed)+'S'+str(alpha)+'alpha.json',orient='table')
        DF_metric.to_json(Address,orient='table')
        df_incident_responded.to_pickle(output_address + '/simulation_example/distance/incident_responded_@' + model_i + '_' + str(num_ambulances) + 'V' + str(average_service_time) + 'h' + str(ambulance_avg_speed) + 'S' + str(alpha) + 'alpha.pkl')
        df_responder_location.to_pickle(output_address + '/simulation_example/distance/responder_location_@' + model_i + '_' + str(num_ambulances) + 'V' + str(average_service_time) + 'h' + str(ambulance_avg_speed) + 'S' + str(alpha) + 'alpha.pkl')
        if Figure_Tag:
            Map=simulation_animation_Plotly(df_incident_responded,df_responder_location)
            if Save_Tag == True:
                Map.write_html(output_address + '/simulation_example/incident&responder' +  model_i + '_' + str(num_ambulances) + 'V' + str(average_service_time) + 'h' + str(ambulance_avg_speed) + 'S' + str(alpha) + '_Map_rate.html')
            else:
                return Map


def Data_collector(metadata):
    '''
    This function collects and generates the data required by simulation. 

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    DF_Test_spacetime : dataframe
        It includes all of the predicted data. The rows are time (x-hour time windows) and space (roadway segments). The columns are the predicted values by each each model. If rolling window is selected for training and test, the prediction of different rollowing windows are appeneded to each other. 
    model_set : list
        a list of the name of the unique models trained.     
    
    df_grouped : dataframe
        it includes the geometry of the roadway segments
    df_incident : dataframe
        a dataframe that inlcudes time, location, and segment that the incidents occured on
    All_seg_incident : dataframe
        a dataframe that includes the unique segments with at least one icidents
    time_range : dataframe
        a dataframe that includes the unique time values

    '''
    
    #1) reading the prediction per time and space for all models

    DF_Test_spacetime=pd.read_pickle(metadata['Input_Address']+'/'+metadata['simulation']['source']+'/DF_likelihood_spacetime.pkl')
    df_unique_seg=DF_Test_spacetime[[metadata['unit_name']]].drop_duplicates()
    
    #2) reading the geometry of the segments and filtering down to the segments we have prediction for
    if metadata['inrix_pickle_address'][-4:] == '.pkl':
        df_grouped = pd.read_pickle(metadata['inrix_pickle_address'])
    else:
        df_grouped = pd.read_parquet(metadata['inrix_pickle_address'])
        uncompatible_list = ['beg', 'end', 'center', 'geometry', 'geometry3d', 'geometry_highres']
        for i in uncompatible_list:
            if i in df_grouped.columns:
                # df[i] = df[i].astype(str)
                from shapely import wkt
                df_grouped[i] = df_grouped[i].apply(wkt.loads)
    #df_grouped=df_grouped.rename(columns={'grouped_type3':metadata['unit_name']})
    df_grouped=df_grouped[df_grouped[metadata['unit_name']].isin(df_unique_seg[metadata['unit_name']])].reset_index().drop('index',axis=1)    
    df_grouped_4326 = (gpd.GeoDataFrame(df_grouped, geometry=df_grouped['geometry'])).copy()
    #df_grouped_4326.plot()
    #All_seg_incident=df_grouped[df_grouped['Grouping'].isin(df_['XDSegID'])].reset_index().drop('index',axis=1)
    All_seg_incident=df_grouped_4326[df_grouped_4326[metadata['unit_name']].isin(df_unique_seg[metadata['unit_name']])].reset_index().drop('index',axis=1)
    All_seg_incident.plot()
    plt.title('Selected Segments')
    
    #3) reading incident data and filtering down to the segments occured on the the segments we have prediction for and time range of interest
    if metadata['incident_pickle_address'][-4:] == '.pkl':
        df_incident = pd.read_pickle(metadata['incident_pickle_address'])
    else:
        df_incident = pd.read_parquet(metadata['incident_pickle_address'])
        from shapely import wkt
        df_incident['geometry'] = df_incident['geometry_string'].apply(wkt.loads) #pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/incident_XDSegID.pkl')
    #df_incident=df_incident.drop('xdsegid',axis=1)
    #df_incident=df_incident.rename(columns={'grouped_type3':metadata['unit_name']})
    if not 'geometry' in df_incident.columns:
        df_incident['geometry'] = df_incident['geometry_string'].apply(shapely.wkt.loads)    
    
    df_incident = df_incident.sort_values('time_local')
    df_incident = df_incident[df_incident['time_local']>=DF_Test_spacetime['time_local'].min()]
    df_incident = df_incident[df_incident['time_local']<=DF_Test_spacetime['time_local'].max()+pd.DateOffset(hours=metadata['window_size']/3600)]
    df_incident = df_incident.reset_index().drop('index',axis=1)
    
    df_incident=df_incident[df_incident[metadata['unit_name']].isin(df_unique_seg[metadata['unit_name']].tolist())].reset_index().drop('index',axis=1)
    df_incident_4326 = (gpd.GeoDataFrame(df_incident, geometry=df_incident['geometry'])).copy()
    df_incident_4326.plot()
    plt.title('Incidents of Selected Segments')
    time_range=DF_Test_spacetime['time_local'].drop_duplicates().sort_values()


    #%%Griding

    
    return DF_Test_spacetime, df_grouped, df_incident, All_seg_incident,time_range
    



def simulation_HyperParams_creator(metadata):
    '''
    This function adjust the hyper parameters required for simulation

    Parameters
    ----------
    metadata : dictionary
        metadata.

    Returns
    -------
    HyperParams : list
        hyper parameters.

    '''
    num_ambulances_list = metadata['simulation']['num_ambulances']
    alpha_list= metadata['simulation']['alpha']
    average_service_time_list=metadata['simulation']['average_service_time']
    ambulance_avg_speed_list=metadata['simulation']['ambulance_avg_speed']
    missed_accident_penalty_list=metadata['simulation']['penalty']
    model_list=metadata['simulation']['model_list']


    HyperParams=[]
    row_id=0
    for num_ambulances in num_ambulances_list:
        for alpha in alpha_list:
            for average_service_time in average_service_time_list:
                for ambulance_avg_speed in ambulance_avg_speed_list:
                    for missed_accident_penalty in missed_accident_penalty_list:
                        for model_i in model_list:
                            HyperParams.append([num_ambulances,alpha, average_service_time,ambulance_avg_speed, missed_accident_penalty ,model_i])
    return HyperParams



def simulation_ExperimentInput_creator(HyperParams, metadata, possible_facility_locations,DF_Test_spacetime,
                                       All_seg_incident,Grid_center, df_incident,demand_nodes, Distant_Dic,time_range):
    """
    

    Parameters
    ----------
    HyperParams : list
        hyper parameters created by simulation_HyperParams_creator.
    metadata : dictionary
        metadata.
    possible_facility_locations : dataframe
        DESCRIPTION.
    DF_Test_spacetime : TYPE
        DESCRIPTION.
    All_seg_incident : TYPE
        DESCRIPTION.
    Grid_center : TYPE
        DESCRIPTION.
    df_incident : TYPE
        DESCRIPTION.
    demand_nodes : TYPE
        DESCRIPTION.
    Distant_Dic : TYPE
        DESCRIPTION.
    time_range : TYPE
        DESCRIPTION.

    Returns
    -------
    experimental_inputs : TYPE
        DESCRIPTION.

    """
    
    '''
    This function creates the inputs required for simulation

    Parameters
    ----------
    metadata : dictionary
        metadata.

    Returns
    -------
    HyperParams : list
        hyper parameters.

    '''   
    experimental_inputs = []
    for num_ambulances,alpha, average_service_time,ambulance_avg_speed, missed_accident_penalty ,model_i in HyperParams:
        # print(num_ambulances, alpha, model_i)
        input_array = [num_ambulances,
                       alpha,
                       model_i,
                       DF_Test_spacetime,
                       All_seg_incident,
                       Grid_center,
                       df_incident,
                       average_service_time,
                       ambulance_avg_speed,
                       missed_accident_penalty,
                       metadata['window_size']/3600  ,
                       possible_facility_locations,
                       demand_nodes,
                       Distant_Dic,
                       time_range,
                       metadata['Output_Address'],
                       metadata['unit_name'],
                       metadata['figure_tag'],
                       ]
        
        experimental_inputs.append(input_array)
    return experimental_inputs    
