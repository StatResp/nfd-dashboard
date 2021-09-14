# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:50:06 2021

@author: Sayyed Mohsen Vazirizade
This code reads the output results of allocation and distance evalution from different models, and put all of them in one DF.
Also it draws a barchart graph and genrate excel tables for the summary of the results. 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
    '''
    axes[0,0].set_ylim(21,31)   
    axes[0,2].set_ylim(20,30)   
    axes[0,2].set_ylim(7,17)
    axes[0,3].set_ylim(5,15)
    
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
    plt.savefig(metadata['Output_Address']+'/simulation/alpha.png')



def simulation_animation_Plotly_2figures(gdf_incident_4326, gdf_responders_4326):
    gdf_incident_4326['type']='Accident'
    gdf_incident_4326['incident_id_str'] = gdf_incident_4326['incident_id'].apply(str)
    gdf_incident_4326['responded_by_str'] = gdf_incident_4326['responded_by'].apply(str)
    gdf_incident_4326['Responded_by_distance_str'] = gdf_incident_4326['Responded_by_distance'].apply(str)
    gdf_incident_4326['start_time_str'] = gdf_incident_4326.start_time.apply(str)
    gdf_incident_4326['time_local_str'] = gdf_incident_4326.time_local.apply(str)
    gdf_incident_4326 = gdf_incident_4326.sort_values('start_time')

    Title = ['Incidents and Responders']

    fig1 = px.scatter_mapbox(gdf_incident_4326, animation_frame='start_time_str',
                             hover_data=['type','incident_id_str','start_time_str','time_local_str','responded_by_str','Responded_by_distance_str'],
                             lat="gps_coordinate_latitude", lon="gps_coordinate_longitude",
                             #color='rgb(10, 10, 172)',
                             mapbox_style='carto-darkmatter',  # mapbox_style="open-street-map",
                             title=Title[0],
                             #color_continuous_scale=px.colors.sequential.Hot,
                             color_discrete_sequence=["red"],
                             center=dict(lat=gdf_incident_4326['gps_coordinate_latitude'].median(), lon=gdf_incident_4326['gps_coordinate_longitude'].median()), size_max=15,
                             zoom=7, range_color=[0, 1])

    gdf_responders_4326['type'] = 'Responder (Ambulance)'
    gdf_responders_4326['ID_str'] = gdf_responders_4326['ID'].apply(str)
    gdf_responders_4326['Grid_ID_str'] = gdf_responders_4326['Grid_ID'].apply(str)
    gdf_responders_4326['Total_Num_Incidents_str'] = gdf_responders_4326['Total_Num_Incidents'].apply(str)
    gdf_responders_4326['start_time_str'] = gdf_responders_4326.start_time.apply(str)
    gdf_responders_4326 = gdf_responders_4326.sort_values('start_time')
    # https://community.plotly.com/t/how-do-i-combine-multiple-animated-plotly-express-scatter-mapbox-traces-on-a-map/50392
    # the problem is the animation for the second plot does not work.
    fig2 = px.scatter_mapbox(gdf_responders_4326,animation_frame='start_time_str',
                             hover_data=['type','ID_str','Grid_ID_str','start_time_str','Total_Num_Incidents_str'],
                             lat="lat", lon="lon",
                             color_discrete_sequence=["blue"],
                             #color="blue",
                            )
    fig1.add_trace(fig2.data[0])
    #fig1.write_html('test.html')
    return fig1

def simulation_animation_Plotly(gdf_incident_4326, gdf_responders_4326):

    #gdf_incident_4326=df_incident_responded.copy()
    #gdf_responders_4326=df_responder_location.copy()
    gdf_incident_4326=gdf_incident_4326[['incident_id','gps_coordinate_latitude', 'gps_coordinate_longitude','time_local','responded_by', 'start_time']].rename(
        columns={'gps_coordinate_latitude':'lat','gps_coordinate_longitude':'lon','incident_id':'ID','time_local':'time_occured'})
    gdf_incident_4326['Type']='Incident'
    gdf_responders_4326=gdf_responders_4326[['ID','lon', 'lat', 'start_time']]
    gdf_responders_4326['Type']='Responder'
    DF=gdf_incident_4326.append(gdf_responders_4326, ignore_index=True)

    DF['ID'] = DF['ID'].apply(str)
    DF['responded_by'] = DF['responded_by'].apply(str)
    DF['start_time_str'] = DF.start_time.apply(str)
    DF['time_occured'] = DF.time_occured.apply(str)
    DF['Type_str'] = DF.Type.apply(str)
    DF = DF.sort_values('start_time')

    Title = ['Incidents and Responders']

    fig1 = px.scatter_mapbox(DF, animation_frame='start_time_str', color='Type',
                             hover_data=['Type_str','ID','start_time_str','time_occured','responded_by'],
                             lat="lat", lon="lon",
                             #color='rgb(10, 10, 172)',
                             mapbox_style='carto-darkmatter',  # mapbox_style="open-street-map",
                             title=Title[0],
                             #color_continuous_scale=px.colors.sequential.Hot,
                             color_discrete_sequence=["red","blue"],
                             center=dict(lat=gdf_incident_4326['lat'].median(), lon=gdf_incident_4326['lon'].median()), size_max=15,
                             zoom=7, range_color=[0, 1])

    #fig1.write_html('test.html')
    return fig1