# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:59:01 2021

@author: Sayyed Mohsen Vazirizade
"""

from pprint import pprint


from statresp.simulation.allocation.Griding_TN import MyGrouping_Grid, Distance_Dict_Builder,Distance_Dict_Builder
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import shapely.geometry as sg
import swifter
from shapely.ops import nearest_points
import time
import folium
from matplotlib import cm
import plotly.express as px




class Accident_Dispatch:
    def __init__(self,df_incident, df_responders,start_time, end_time):
            #self.A = a
            #Read the Incident DF
            '''
            #self.df_incident = pickle.load(open(metadata['incidentdf_pickle_address'], 'rb'))
            self.df_incident =pickle.load(open('sample_data/incident_XDSegID.pkl', 'rb'))
            self.df_incident =self.df_incident.sort_values('time_local')
            '''
            self.df_incident=df_incident
            #self.df_incident=self.df_incident[(self.df_incident['time_local']>=metadata['start_time_test']) & (self.df_incident['time_local']<=metadata['end_time_test']) ]
            self.df_incident=self.df_incident[(self.df_incident['time_local']>=start_time) & (self.df_incident['time_local']<end_time) ]
            self.df_incident=self.df_incident[self.df_incident['dist_to_seg'].notna()]
            #Read the Dispatcher
            #self.df_responders=pd.DataFrame({'ID':[1,2,3], 'lat':[35,35.5,36],'lon':[-90,-86,-82]})
            self.df_responders=df_responders
            
            
            #Convert both to DF    
            self.gdf_incident_4326=self._GDF_maker(self.df_incident)
            self.gdf_responders_4326=self._GDF_maker( self.df_responders)
            
            
            if len(self.gdf_incident_4326)>0:
                #Finding the closest Dispatcher to each accident
                gdf_incident_3310 = self.gdf_incident_4326.to_crs(epsg = 3310)
                gdf_responders_3310 = self.gdf_responders_4326.to_crs(epsg = 3310)
                
                self.gdf_incident_4326["Nearest_responder"],self.gdf_incident_4326["Nearest_Responder_distance"]  = zip(*gdf_incident_3310.apply(self._calculate_nearest, destination=gdf_responders_3310, axis=1))
                #print('printing gdf_incident_4326',self.gdf_incident_4326)

                DF=self.gdf_incident_4326[["Nearest_responder",'incident_id']].groupby('Nearest_responder').count()
                DF=DF.reset_index().rename(columns={'incident_id':'Total_Num_Incidents','Nearest_responder':'ID'})
                self.gdf_responders_4326=pd.merge(self.gdf_responders_4326,DF, left_on='ID', right_on='ID', how='left' )            
                
                
                
    def _GDF_maker(self, DF):
        #Convert DF to GDF using 'epsg:4326'
        
        crs = {'init': 'epsg:4326'}
        DF_4326 = DF.copy()
        if 'geometry' in DF.columns: 
                DF_4326 = gpd.GeoDataFrame(DF_4326, geometry=DF_4326['geometry'], crs=crs)
        else:
                Geom=DF_4326.apply(lambda ROW: sg.Point(ROW['lon'], ROW['lat']), axis=1)
                DF_4326 = gpd.GeoDataFrame(DF_4326, geometry=Geom, crs=crs)
        return DF_4326
                
        
        
    def _calculate_nearest(self, row, destination):
        """
        it calulates the distacne from a shape (all weather stations) to a point (beggining of each segment) and find the closest shape id
        @param row: a row of a pandas dataframe (beggining of each segment) that we want to find the closest station to 
        @param destination: a gepandas datafrane including the shapes (all weather stations)
        @return: match_value:   the closest station_id to the row
        """
        # 1 - create unary union    
        dest_unary = destination["geometry"].unary_union
        # 2 - find closest point
        nearest_geom = nearest_points(row["geometry"], dest_unary)
        # 3 - Find the corresponding geom
        match_geom = destination.loc[destination.geometry == nearest_geom[1]]
        # 4 - get the corresponding value
        match_value = match_geom["ID"].to_numpy()[0]
        # 5 - get the distance between the intented point and the closest point to in in the distination 
        Distance_km=nearest_geom[0].distance(nearest_geom[1])/1000
        return match_value, Distance_km



    def Plot(self):
        
        gdf_incident_4326=self.gdf_incident_4326
        gdf_responders_4326=self.gdf_responders_4326
        plot_area= gpd.GeoDataFrame(data=gdf_incident_4326[['incident_id','geometry']])
        plot_area = plot_area.to_json()
        Map = folium.Map(location = [36.151, -86.852], opacity=0.5, tiles='cartodbdark_matter', zoom_start = 6)   #'Stamen Toner'
        #title_html = '''     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '''.format(time_range[0].strftime('%Y-%m-%d %H:%M:%S %Z'))
        cmap = cm.get_cmap('tab20', 20)  
        #colordict={i/2: 'rgb'+str((int(cmap(i)[0]*255),int(cmap(i)[1]*255),int(cmap(i)[2]*255))) for i in range(0,20,2)}
        #colordict_2={(i-1)/2+10: 'rgb'+str((int(cmap(i)[0]*255),int(cmap(i)[1]*255),int(cmap(i)[2]*255))) for i in range(1,20,2)}
        #colordict.update(colordict_2)
        colordict={i: 'rgb'+str((int(cmap(i)[0]*255),int(cmap(i)[1]*255),int(cmap(i)[2]*255))) for i in range(0,20,1)}
        #'feature.properties.unit_segment_id'

        for row,row_values  in gdf_incident_4326.iterrows():
            #location = [row_values['geometry'].y, row_values['geometry'].x]
            #popup = str(row_values['Incident_ID'])+ ': ' + '\n Nearest_responder:' + str(row_values['Nearest_responder'])+'M'
            #marker = folium.Marker(location = location, popup = popup)
            #marker.add_to(Map)
            folium.Circle(
                location = [row_values['geometry'].y, row_values['geometry'].x],
                #radius=np.nan_to_num(row_values['district_pop_2001'])**0.5*4,
                radius=1,
                #color=colordict[row_values['Nearest_responder']]
                color='red'
            #marker = folium.Marker(location = location, popup = popup)
            ).add_to(Map)


        for row, row_values in gdf_responders_4326.iterrows():
            '''
            location = [row_values['geometry'].y, row_values['geometry'].x]
            popup = str(row_values['ID'])+ ': ' + '\n Dependant Points:' + str(row_values['ID'])
            marker = folium.Marker(location = location, popup = popup, color=[row_values['ID']]).add_to(Map)
            '''

            folium.Circle(
                location=[row_values['geometry'].y, row_values['geometry'].x],
                # radius=np.nan_to_num(row_values['district_pop_2001'])**0.5*4,
                radius=10000,
                color=colordict[row_values['ID']],
                fill=True,
                fill_color=colordict[row_values['ID']],
                # marker = folium.Marker(location = location, popup = popup)
            ).add_to(Map)
            # Map.save('test2.html')

        return (Map)


    def Plotly(self):

        gdf_incident_4326 = self.gdf_incident_4326
        gdf_incident_4326['incident_id_str'] = gdf_incident_4326['incident_id'].apply(str)
        gdf_incident_4326['logmle_str'] = gdf_incident_4326['logmle'].apply(str)
        gdf_incident_4326['county_incident_str'] = gdf_incident_4326['county_incident'].apply(str)
        gdf_incident_4326['route_str'] = gdf_incident_4326['route'].apply(str)
        gdf_incident_4326['responded_by_str'] = gdf_incident_4326['responded_by'].apply(str)
        gdf_incident_4326['Responded_by_distance_str'] = gdf_incident_4326['Responded_by_distance'].apply(str)

        gdf_incident_4326 = gdf_incident_4326.sort_values('time_local')
        gdf_incident_4326['time'] = gdf_incident_4326.time_local.apply(str)

        Title = ['Incidents and Responders']

        fig1 = px.scatter_mapbox(gdf_incident_4326, #animation_frame='time', animation_group='index',
                                 hover_data=['incident_id_str','time','logmle_str','county_incident_str','route_str','responded_by_str','Responded_by_distance_str'],
                                 lat="gps_coordinate_latitude", lon="gps_coordinate_longitude",
                                 #color='rgb(10, 10, 172)',
                                 mapbox_style='carto-darkmatter',  # mapbox_style="open-street-map",
                                 title=Title[0],
                                 #color_continuous_scale=px.colors.sequential.Hot,
                                 color_discrete_sequence=["red"],
                                 center=dict(lat=gdf_incident_4326['gps_coordinate_latitude'].median(), lon=gdf_incident_4326['gps_coordinate_longitude'].median()), size_max=15,
                                 zoom=7, range_color=[0, 1])




        gdf_responders_4326 = self.gdf_responders_4326
        gdf_responders_4326['ID_str'] = gdf_responders_4326['ID'].apply(str)
        gdf_responders_4326['Grid_ID_str'] = gdf_responders_4326['Grid_ID'].apply(str)
        gdf_responders_4326['Total_Num_Incidents_str'] = gdf_responders_4326['Total_Num_Incidents'].apply(str)
        fig2 = px.scatter_mapbox(gdf_responders_4326,# animation_frame='time', animation_group='index',
                                 hover_data=['ID_str','Grid_ID_str','Total_Num_Incidents_str'],
                                 lat="lat", lon="lon",
                                 #color="blue",
                                )
        fig1.add_trace(fig2.data[0])
        #fig1.write_html('test.html')
        return fig1


    
    
    
    def Responder_Finder(self, start_time,  average_service_time, ambulance_avg_speed):
    
        gdf_incident_3310   = self.gdf_incident_4326.to_crs(epsg = 3310)
        gdf_responders_3310 = self.gdf_responders_4326.to_crs(epsg = 3310)
        gdf_responders_3310['time_local_available']=start_time
        
        #accident=gdf_incident_3310.iloc[0]
        for row_index, accident in gdf_incident_3310.iterrows():
            #print('\n')
        #accident=gdf_incident_3310.iloc[3]
            #print(accident[['time_local','Nearest_responder','Nearest_Responder_distance']])
            List_Available_Responders=gdf_responders_3310[gdf_responders_3310['time_local_available']<=accident['time_local']]
            #print(List_Available_Responders[['ID','time_local_available']])
            if len(List_Available_Responders)>0:
                if  (accident['Nearest_responder'] in (List_Available_Responders['ID'].tolist()))==False:
                    #print('search for new')
                    accident["Nearest_responder"],accident["Nearest_Responder_distance"] =self._calculate_nearest(accident, List_Available_Responders)
                #elif (accident['Nearest_responder'] in (List_Available_Responders['ID'].tolist()))==True:
                #else:
                   #print('already exist')
                Mask=gdf_responders_3310['ID']==accident['Nearest_responder']
                gdf_responders_3310.loc[Mask,'time_local_available']=accident['time_local']+pd.DateOffset(hours=accident["Nearest_Responder_distance"]/ambulance_avg_speed)+pd.DateOffset(hours=average_service_time)
                self.gdf_incident_4326.loc[row_index,'responded_by']=accident['Nearest_responder']
                self.gdf_incident_4326.loc[row_index,'Responded_by_distance']=accident['Nearest_Responder_distance']
                #self.gdf_incident_4326.loc[row_index,'WastedTime_Responder']=accident["Nearest_Responder_distance"]/ambulance_avg_speed+average_service_time
                #print('printing gdf_incident_4326 \n', self.gdf_incident_4326[['incident_id', 'gps_coordinate_latitude', 'gps_coordinate_longitude', 'time_local', 'responded_by', 'Responded_by_distance']])
                #print(self.gdf_incident_4326.columns)

            else:
                #print('no available responder found!')
                pass




def Dispaching_Scoring(ROW_DF_Test_metric_time,df_responders,df_incident ,window_size_hour,average_service_time, ambulance_avg_speed,missed_accident_penalty=None,model_i=None,alpha=None, Figure_Tag=False):
        start_time=ROW_DF_Test_metric_time
        end_time=start_time+pd.DateOffset(hours=window_size_hour)   
        Dispatch_0=Accident_Dispatch(df_incident,df_responders, start_time,end_time)
        Dispatch_0.Responder_Finder(start_time,average_service_time,ambulance_avg_speed)
        #Dispatch_0.df_responders
        #Dispatch_0.gdf_responders_4326
        #ROW_DF_Test_metric_time_added= pd.Series(dtype=float)
        #Tag='@'+model_i+'_'+str(len(df_responders))+'V'+str(average_service_time)+'h'+str(alpha)+'alpha'
        Tag=''
        Dispatch_Output={}
        if len(Dispatch_0.gdf_incident_4326)>0:
            Dispatch_Output['DistanceTravel'+Tag]=Dispatch_0.gdf_incident_4326['Responded_by_distance'].fillna(missed_accident_penalty, inplace = False).sum()
            Dispatch_Output['TotalNumAccidents'+Tag]=len(Dispatch_0.gdf_incident_4326['responded_by'])
            Dispatch_Output['TotalNumAccidentsNotResponded'+Tag]=sum(Dispatch_0.gdf_incident_4326['responded_by'].isna())
            Dispatch_Output['DistanceTravelPerAccident'+Tag]=Dispatch_Output['DistanceTravel'+Tag]/(Dispatch_Output['TotalNumAccidents'+Tag]-Dispatch_Output['TotalNumAccidentsNotResponded'+Tag])
        else:
            #print('No Accident occurs')
            Dispatch_Output['DistanceTravel'+Tag]=0
            Dispatch_Output['TotalNumAccidents'+Tag]=0
            Dispatch_Output['TotalNumAccidentsNotResponded'+Tag]=0
            Dispatch_Output['DistanceTravelPerAccident'+Tag]=0
        '''
        if Figure_Tag==True:
            #Map=Dispatch_0.Plot()
            ##Map.save('output/'+'accidents_1'+start_time.strftime('%Y-%m-%d %H')+'_Map_rate.html')
            Map=Dispatch_0.Plotly()
            Map.write_html('output/'+'accidents_1'+start_time.strftime('%Y-%m-%d %H')+'_Map_rate.html')
            print('map is generated: ','output/'+'accidents_allocation'+model_i+start_time.strftime('%Y-%m-%d %H')+'_Map_rate.html')
        '''

        if len(Dispatch_0.gdf_incident_4326) > 0:
            Dispatch_0.gdf_incident_4326['start_time']=start_time
            Dispatch_0.gdf_responders_4326['start_time'] = start_time
            return Dispatch_Output, Dispatch_0.gdf_incident_4326[['incident_id', 'gps_coordinate_latitude', 'gps_coordinate_longitude', 'time_local', 'responded_by', 'Responded_by_distance','start_time']], Dispatch_0.gdf_responders_4326
        else:
            return Dispatch_Output, pd.DataFrame(), pd.DataFrame()
    
    
    
#%%

def Find_DemandSupply_Location(df_inrix=None,df_incident=None,All_seg_incident=None,metadata=None,width = 0.1,height = 0.1 ,Source_crs='EPSG:4326', Intended_crs='EPSG:4326'    ):
    '''
    This function prepares the input for the optimization engine which includes the distance matrix and location of each grid and segments

    Parameters
    ----------
    df_inrix : dataframe
        A dataframe that includes the geometry of the segments
    df_incident : dataframe
        A dataframe that includes time and coordinates of the incidents
    All_seg_incident : dataframe
        A dataframe that just includes the segments of interest
    width and height: float
        This defines the size of the grids. If  Intended_crs='EPSG:4326' they are degrees, and if Intended_crs='EPSG:3310' is selected, they are meters.

    Returns
    -------
    possible_facility_locations : set
        DESCRIPTION.
    demand_nodes : set
        DESCRIPTION.
    Distant_Dic : dictionary
        A dictionary that contains the distance between each demand node and possible facility location.
    All_seg_incident : dataframe
        A dataframe that just includes the segments of interest
    Grid_center : dataframe
        a dataframe that contains the geometry of the center of the grids

    '''
    Grid, Grid_center=MyGrouping_Grid(df_inrix,df_incident,metadata ,width,height,Source_crs, Intended_crs)
    Distant_Dic,All_seg_incident=Distance_Dict_Builder(metadata,All_seg_incident,Grid_center)
    #All_seg_incident=All_seg_incident.rename(columns={'grouped_type3':metadata['unit_name']})
    
    possible_facility_locations = set(Grid_center['Grid_ID'])  #you can put filter on this
    demand_nodes = set(All_seg_incident[metadata['unit_name']])    
    
    
    return possible_facility_locations,demand_nodes,Distant_Dic,All_seg_incident,Grid_center


def Weight_and_Merge(DF_Test_spacetime,All_seg_incident,time_i,unit_name,model='uniform'):
    '''
    This function caluculate the weighted distance based in the predicted likelihood.

    Parameters
    ----------
    DF_Test_spacetime : DF
        This includes the prediction infromation for each time frame and segment.
    All_seg_incident : DF
        This includes all the segments; top 20% groups
    time_i : TYPE
        DESCRIPTION.
    model : TYPE, optional
        DESCRIPTION. The default is 'uniform'. Choose 'unifrom', if you dont want to predict anything and you want to consider all segments equally dangerous.


    Returns
    -------
    weights_dict : TYPE
        DESCRIPTION.
    DF_Test_space_time_i : TYPE
        DESCRIPTION.

    '''
    DF_Test_space_time_i=pd.merge(DF_Test_spacetime[DF_Test_spacetime['time_local']==time_i], All_seg_incident[[unit_name,'geometry','line','center']],left_on=unit_name, right_on=unit_name, how='right')
    DF_Test_space_time_i.fillna(0, inplace = True) 
    weights_dict=dict()
    if model=='uniform':
        #weights_dict=(DF_Test_space_time_i['XDSegID']*0+1).to_dict()
        weights_dict=dict(zip(DF_Test_space_time_i[unit_name], (DF_Test_space_time_i[unit_name]*0+1)))
    else:
        #weights_dict=DF_Test_space_time_i[model].to_dict()
        weights_dict=dict(zip(DF_Test_space_time_i[unit_name], DF_Test_space_time_i[model]))
    '''    
    for n_l in demand_nodes:
            if model=='uniform':
                weights_dict[n_l] =1
            else:
                weights_dict[n_l] =DF_Test_space_time_i[DF_Test_space_time_i['XDSegID']==n_l][model].iloc[0]  
    '''

    return weights_dict, DF_Test_space_time_i



def Responders_Location(Grid_center,Responders_GridID,DF_Test_space_time_i,time_i=None,model='uniform',alpha=None, Figure_Tag=False):
    '''
    This function finds the location of responders based on the center of the grid they belong to.

    Parameters
    ----------
    DF_Test_space_time_i : DF
        DESCRIPTION.
    Grid_center : GDF
        DESCRIPTION.
    Responders_GridID : Dic
        DESCRIPTION.
    model : TYPE, optional
        DESCRIPTION. The default is 'uniform'.

    Returns
    -------
    df_responders : TYPE
        DESCRIPTION.

    '''

    df_responders        = Grid_center[Grid_center['Grid_ID'].isin(Responders_GridID)].copy()
    df_responders['lon'] = df_responders.apply(lambda row: row['geometry'].coords[0][0],axis=1)
    df_responders['lat'] = df_responders.apply(lambda row: row['geometry'].coords[0][1],axis=1)
    df_responders        = df_responders.reset_index().reset_index().drop('index', axis=1).rename(columns={'level_0':'ID'})
    
    if Figure_Tag==True:
        fig, axis = plt.subplots(1,1,figsize=(10,5))
        DF_Test_space_time_i_4326=gpd.GeoDataFrame(DF_Test_space_time_i, geometry=DF_Test_space_time_i['line'], crs={'init': 'epsg:4326'} )
        
        if model=='uniform':
            DF_Test_space_time_i_4326.plot(legend=False,ax=axis)
        else:
            DF_Test_space_time_i_4326.plot(DF_Test_space_time_i_4326[model],legend=True,ax=axis)
        df_responders.plot(ax=axis)
        axis.set_title(time_i.strftime('%Y-%m-%d %H:%M:%S %Z')+', alpha='+str(alpha)+', ' + model)
        fig.show()
    return df_responders


def Graph_Distance_Metric(DF_Test_metric_time,Title):
    '''
    This function is used just for plotting of different metrics.
    '''
    #print(DF)

    #plt.figure(figsize=[20,10])
    fig, axis = plt.subplots(4,1,figsize=(15,15))
    sns.set_palette("tab10")
    DF_summary_=DF_Test_metric_time.set_index('time_local')
    metric_list=['RespondedbyDistance','TotalNumberAccidents','TotalNumberAccidentsNotResponded','RespondedbyDistanceperAccdient']
    for i in range(4):
        DF=DF_summary_[[j for j in DF_summary_.columns if j.split('_')[-1].split('?')[0]==metric_list[i]]]
        ax=axis[i]
        DF.plot(linewidth=1,ax=ax) 
        ax.set_xlabel('')
        ax.set_ylabel(metric_list[i])
        #ax.set_title(metric_list[i])
        ax.set_xticks([],minor=False)   #position
        

        for i,Value in enumerate(DF.mean(axis=0)):
            ax.axhline(Value, ls='--',linewidth=2,color=sns.color_palette("tab10")[i])
    ax.set_xlabel('Time (Local)')
    ax.set_xticks(DF.index[::6],minor=False)   #position
    ax.set_xticklabels(DF.index[::6], rotation=90,minor=False) 	  #label rotation, etc. 
    ax.set_xticks([],minor=True)   #position
    axis[0].set_title(Title)    
    plt.show()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    