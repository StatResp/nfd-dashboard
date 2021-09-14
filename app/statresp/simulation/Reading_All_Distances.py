# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:50:06 2021

@author: Sayyed Mohsen Vazirizade
This code reads the output results of allocation and distance evalution from different models, and put all of them in one DF.
Also it draws a barchart graph and genrate excel tables for the summary of the results. 
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_performance_simulation(HyperParams,metadata):
    '''
    This function collects the json files created by simulation parallel processing module and saved in simulation/distance and saves that. 

    Parameters
    ----------
    HyperParams : list
        HyperParams.
    metadata : dictionary
        metadata.

    Returns
    -------
    DF_metric_allmethod_time : dataframe
        This includes the output of simulation for evaluating the performance of each model for each hyperparameter during each time window. The metrics are the total travel distance of the responders,  the total travel distance of the responders per incident, the total number of incidents left unattended due to the unavailability of resources, and the total number of incidents during that time window.  .

    '''
    
    model_list=metadata['simulation']['model_list']
    #DF_metric_all=pd.DataFrame(columns=['num_ambulances','alpha','model_i','DistanceTravel','TotalNumAccidents','TotalNumAccidentsNotResponded','DistanceTravelPerAccident'],index=range(len(num_ambulances)*len(alpha_list)*len(model_list)))
    
    DF_metric_allmethod_time=pd.DataFrame(columns=['num_ambulances','alpha','average_service_time','ambulance_avg_speed','missed_accident_penalty','model_i','DistanceTravel','TotalNumAccidents','TotalNumAccidentsNotResponded','DistanceTravelPerAccident'])
    
    row_id=0
    for num_ambulances,alpha, average_service_time,ambulance_avg_speed, missed_accident_penalty ,model_i in HyperParams:
        DF_temp=pd.read_json(metadata['Output_Address']+'/simulation_example/distance/Distance-Metric_@'+model_i+'_'+str(num_ambulances)+'V'+str(average_service_time)+'h'+str(ambulance_avg_speed)+'S'+str(alpha)+'alpha.json',orient='table')
        DF_temp['num_ambulances']=num_ambulances
        DF_temp['alpha']=alpha
        DF_temp['average_service_time']=average_service_time
        DF_temp['ambulance_avg_speed']=ambulance_avg_speed
        DF_temp['missed_accident_penalty']=missed_accident_penalty
        DF_temp['model_i']=model_i
        DF_temp['model_index']=model_list.index(model_i)
        #DF_metric_all.loc[row_id, ['DistanceTravel','TotalNumAccidents','TotalNumAccidentsNotResponded','DistanceTravelPerAccident']]=A
        DF_metric_allmethod_time=DF_metric_allmethod_time.append(DF_temp)
        row_id+=1
    
    DF_metric_allmethod_time=DF_metric_allmethod_time.reset_index().drop('index', axis=1)
    
    #print(DF_metric_allmethod_time)
    NAME='All'
    DF_metric_allmethod_time.to_pickle(metadata['Output_Address']+'/simulation_example/Distance_'+NAME+'.pkl')
    #DF_metric_allmethod_time=pd.read_pickle(metadata['Output_Address']+'/simulation/Distance_All.pkl')
    print('Reading simulation results is done.')
    return DF_metric_allmethod_time

#%% 
#DF_metric_allmethod_time=pickle.load(open('results/Distance_LR+NN+RF_Jan.pkl', 'rb'))
#Building the table using the mean of all:
    
def average_performance_simulation(DF_metric_allmethod_time,metadata):
    '''
    This function the summerizes the output of the simulation module, DF_metric_allmethod_time in an easy-to-read format. It saves it in a excel file with four tabs, DistanceTravel_mean,  TotalNumAccNotResp_mean, TotalNumAccNotResp_max, and DistanceTravelPerAcc_mean. It provides user-friendly comparision tables of different models and hyperparameters.  

    Parameters
    ----------
    DF_metric_allmethod_time : TYPE
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    DF_Distance_DistanceTravel_mean : dataframe
        DESCRIPTION.
    DF_NotResponded_TotalNumAccidentsNotResponded_mean : dataframe
        DESCRIPTION.
    DF_NotResponded_TotalNumAccidentsNotResponded_max : dataframe
        DESCRIPTION.
    DF_DistancePerAccident_DistanceTravelPerAccident_mean : dataframe
        DESCRIPTION.

    '''
    NAME='All'
    model_list=metadata['simulation']['model_list']
    DF_metric_all_mean=DF_metric_allmethod_time.groupby(['model_i','alpha','num_ambulances','model_index']).mean().reset_index().sort_values(['model_index','num_ambulances','alpha']).reset_index().drop('index',axis=1)
    DF_metric_all_max=DF_metric_allmethod_time.groupby(['model_i','alpha','num_ambulances','model_index']).max().reset_index().sort_values(['model_index','num_ambulances','alpha']).reset_index().drop('index',axis=1)
    
    DF_Distance_DistanceTravel_mean=DF_metric_all_mean.pivot(index='model_i', columns=['num_ambulances','alpha'])['DistanceTravel'].loc[model_list].reset_index()
    
    DF_NotResponded_TotalNumAccidentsNotResponded_mean=DF_metric_all_mean.pivot(index='model_i', columns=['num_ambulances','alpha'])['TotalNumAccidentsNotResponded'].loc[model_list].reset_index()
    DF_NotResponded_TotalNumAccidentsNotResponded_max=DF_metric_all_max.pivot(index='model_i', columns=['num_ambulances','alpha'])['TotalNumAccidentsNotResponded'].loc[model_list].reset_index()
    
    DF_DistancePerAccident_DistanceTravelPerAccident_mean=DF_metric_all_mean.pivot(index='model_i', columns=['num_ambulances','alpha'])['DistanceTravelPerAccident'].loc[model_list].reset_index()
    
    with pd.ExcelWriter(metadata['Output_Address']+'/simulation_example/AllocationResults_'+NAME+'.xlsx') as writer:
        DF_Distance_DistanceTravel_mean.to_excel(writer, sheet_name='DistanceTravel_mean')
        DF_NotResponded_TotalNumAccidentsNotResponded_mean.to_excel(writer, sheet_name='TotalNumAccNotResp_mean')
        DF_NotResponded_TotalNumAccidentsNotResponded_max.to_excel(writer, sheet_name='TotalNumAccNotResp_max')
        DF_DistancePerAccident_DistanceTravelPerAccident_mean.to_excel(writer, sheet_name='DistanceTravelPerAcc_mean')
    return DF_Distance_DistanceTravel_mean, \
           DF_NotResponded_TotalNumAccidentsNotResponded_mean,            \
           DF_NotResponded_TotalNumAccidentsNotResponded_max, \
           DF_DistancePerAccident_DistanceTravelPerAccident_mean
#%% 
#Building the bar chart:
def Box_plot(DF_metric_allmethod_time, y,num_ambulances, Log_Tag ):
    DF_metric_allmethod_time['alpha ']='alpha = '+DF_metric_allmethod_time['alpha'].astype(str)
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(figsize=(20,15))
    
    
#    DF_metric_allmethod_time['alpha ']='alpha='+DF_metric_allmethod_time['alpha'].astype(str)
#    fig=sns.barplot(x = 'model_i', y = 'DistanceTravel', hue='alpha ',  data = DF_metric_allmethod_time[DF_metric_allmethod_time['num_ambulances']==num_ambulances],
#                palette = 'hls', ci = 'sd' ,
#                capsize = .1, errwidth = 0.5,
#                ax= ax            )    
    
    fig=sns.boxplot(x = 'model_i', y = y, hue='alpha ',  data = DF_metric_allmethod_time[DF_metric_allmethod_time['num_ambulances']==num_ambulances],
                palette = 'hls',fliersize=1, linewidth=1,whis=100,
                ax= ax  )    
    
    plt.xticks(rotation=90)
    plt.legend(loc=9)
    #plt.yscale('log')
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Travel Distance per Accident (km)')
    ax.set_title('p = '+str(num_ambulances))
    if Log_Tag==True:
        #ax.set(yscale="log");ax.set_ylim(3, 30) 
        ax.set(yscale="log");ax.set_ylim(1, 20) 
    #ax.set_ylim(-220, 1300) 
    #ax.set_ylim(1, 5000)      

def simulation_bar_chart(DF_metric_allmethod_time,metadata):
    for i in metadata['simulation']['num_ambulances']:
        Box_plot(DF_metric_allmethod_time[DF_metric_allmethod_time['TotalNumAccidents']>0], 'DistanceTravelPerAccident',i,Log_Tag=False);plt.savefig(metadata['Output_Address']+'/simulation_example/DistanceTravelPerAccident_P='+str(i)+'.png')
    #Box_plot(DF_metric_allmethod_time[DF_metric_allmethod_time['TotalNumAccidents']>0], 'DistanceTravelPerAccident',5,Log_Tag=False);plt.savefig(metadata['Output_Address']+'/somulation/DistanceTravelPerAccident_P=5.png')
    #Box_plot(DF_metric_allmethod_time[DF_metric_allmethod_time['TotalNumAccidents']>0], 'DistanceTravelPerAccident',10,Log_Tag=False);plt.savefig(metadata['Output_Address']+'/somulationresults/DistanceTravelPerAccident_P=10.png')
    #Box_plot(DF_metric_allmethod_time[DF_metric_allmethod_time['TotalNumAccidents']>0], 'DistanceTravelPerAccident',15,Log_Tag=False);plt.savefig(metadata['Output_Address']+'/somulationresults/DistanceTravelPerAccident_P=15.png')
    #Box_plot(DF_metric_allmethod_time[DF_metric_allmethod_time['TotalNumAccidents']>0], 'DistanceTravelPerAccident',20,Log_Tag=False);plt.savefig(metadata['Output_Address']+'/somulationresults/DistanceTravelPerAccident_P=20.png')
    




#%% Analysis of Alpha

def simulation_alpha_chart(DF_metric_allmethod_time, metadata):
    if len(metadata['simulation']['alpha'])<3:
        print('Not enough values for drawing the alpha chart')
        return 
    DF_metric_all=DF_metric_allmethod_time.copy()
    
    
    DF_metric_all['model_i ']='models'
    DF_metric_all.loc[DF_metric_all['model_i']=='Naive','model_i ']='Naive'
    DF_metric_all_Added=DF_metric_all.groupby(['model_i ','alpha','num_ambulances']).mean().reset_index()
    DF_metric_all_Added=DF_metric_all_Added[DF_metric_all_Added['model_i ']=='models']
    DF_metric_all_Added['model_i']='mean of models'
    DF_metric_all_Added['model_i ']='mean of models'
    DF_metric_all=DF_metric_all.append(DF_metric_all_Added).reset_index().drop('index',axis=1)
    
    
    palette={}
    for i in DF_metric_all['model_i'].unique():
        palette[i]='grey'
    palette['Naive']='red'
    palette['mean of models']='blue'
    sns.set_style('whitegrid')
    
    DF_metric_all=DF_metric_all.rename(columns={'num_ambulances':'Number of the Resources'})
    DF_metric_all=DF_metric_all.rename(columns={'DistanceTravel':'Travel Distance (km)'})
    DF_metric_all=DF_metric_all.rename(columns={'DistanceTravelPerAccident':'Travel Distance per Accident (km)'})
    
    
    sns.set(font_scale=2.5)
    sns.set_style('whitegrid')
    
    fig=sns.lmplot(x='alpha',y='Travel Distance (km)', hue='model_i', data = DF_metric_all,
               ci=None, order=2,scatter=False, line_kws={"lw":1.5}, col='Number of the Resources',sharey=False,
               palette=palette,
               legend=False,truncate=False,
               height=8, aspect=1
               )
    #fig.fig.subplots_adjust(wspace=1)
    plt.xlim(-0.25, 2.5)
    axes = fig.axes
    
    axes[0,0].set_ylim(21,31)   
    axes[0,2].set_ylim(20,30)   
    axes[0,2].set_ylim(7,17)
    axes[0,3].set_ylim(5,15)
    '''
    axes[0,0].set_ylim(26,29)   
    axes[0,2].set_ylim(18,21)   
    axes[0,2].set_ylim(16,19)
    axes[0,3].set_ylim(16,19)
    
    axes[0,0].set_ylim(11,21)   
    axes[0,1].set_ylim(9,19)
    axes[0,2].set_ylim(6,16)
    
    
    axes[0,0].set_ylim(19.25,20.75)   
    axes[0,1].set_ylim(17.25,18.75)
    axes[0,2].set_ylim(17.25,18.75)
    
    axes[0,0].set_ylim(550,700)   
    axes[0,1].set_ylim(275,425)
    axes[0,2].set_ylim(200,350)
    
    axes[0,0].set_ylim(450,600)   
    axes[0,1].set_ylim(250,400)
    axes[0,2].set_ylim(150,300)
    '''
    #plt.ylim(450, 600)
    #fig.set(ylim=(350, None))
    #plt.savefig('results/alpha.png')
    plt.savefig(metadata['Output_Address']+'/simulation_example/alpha_NN.png')

    