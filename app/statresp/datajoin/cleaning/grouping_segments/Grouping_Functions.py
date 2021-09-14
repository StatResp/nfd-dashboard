# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:50:15 2020

@author: vaziris
This scripts is  written by Sayyed Mohsen Vazirizade.  s.m.vazirizade@gmail.com
"""

from pprint import pprint
import pandas as pd
import geopandas as gpd 
import numpy as np
import shapely.geometry as sg
from shapely.geometry import Polygon
from shapely import ops 
from statresp.datajoin.cleaning.utils import Save_DF
import seaborn as sns
import matplotlib.pyplot as plt


def Reshaper(ROW,inrix_df):
    """
    a groups incorporates multiple segments. This function put that segments together and make a shapefile out of them. 
    @param ROW: One row of grouped dataframe which includes the list of the segments it includes and their orders
    @param inrix_df:  the inrix dataframe which includes XDSegID, XDGroup
    @return: merged_line:  the combined geometry
                              
    """    
    #print(ROW)
    List_of_Inrix_Geom=inrix_df[inrix_df['XDSegID'].isin(ROW['Members'])].sort_values('grouped_XDSegID_id')['geometry'].tolist()
    multi_line = sg.MultiLineString(List_of_Inrix_Geom)
    merged_line = ops.linemerge(multi_line)
    return merged_line





def MyGrouping_XD(TestDF):
    """
    it groups the segments together using grouping method 1
    In grouping method 1, we use 'XDGroup' to put the segments together. 
    @param TestDF: the inrix dataframe which includes XDSegID, XDGroup, and Miles
    @return: DF_grouped:   it is a dataframe that summerized the groups, the length in miles, the size (the total number of the segments), and the list of the members                               
    """
    DF_grouped=TestDF[['XDSegID','Miles','XDGroup']].groupby('XDGroup').agg({'XDSegID': ['count'], 'Miles': ['sum']}) 
    DF_grouped.columns=['Size','Miles']
    DF_grouped=DF_grouped.reset_index()
    DF_grouped=DF_grouped.rename(columns={'XDGroup':'Grouping'})
    DF_grouped['Members']=  DF_grouped.apply(lambda ROW: TestDF[TestDF['XDGroup']==ROW['Grouping']]['XDSegID'].tolist(),axis=1) 
    return DF_grouped


    

def MyGrouping_type_1(TestDF):
    #this function just put the segments back to back based on the XDSegID, PreviousXD, NextXDSegI
    """
    it groups the segments together using grouping method 1
    In grouping method 1, we use 'XDGroup' to put the segments together. 
    @param TestDF: the inrix dataframe which includes XDSegID, XDGroup, and Miles
    @return: DF_grouped:   it is a dataframe that summerized the groups, the length in miles, the size (the total number of the segments), and the list of the members
    @return: TestDF:   it is a dataframe including the orginal TestDF dataframe and more columns regarding the the group id and the member id                           
    """
    TestDF['MyGrouping_1']=0
    #TestDFBeg=TestDF[((TestDF['NextXDSegI'].notna()) & (TestDF['PreviousXD'].isna())) | (FRC0['PreviousXD'].isin(FRC0['XDSegID'].unique())==False)]
    TestDFBeg=TestDF[(TestDF['PreviousXD'].isna()) | (TestDF['PreviousXD'].isin(TestDF['XDSegID'].unique())==False)] 
    MyGrouping_1_i=1000000
    for ID in range(len(TestDFBeg)):
        Name=TestDFBeg.iloc[ID]['XDSegID']
        id=TestDF['XDSegID']==Name
        MyGrouping_1_i+=1
        TestDF.loc[id,'MyGrouping_1']=MyGrouping_1_i
        MyGrouping_1_id_i=1
        TestDF.loc[id,'MyGrouping_1_id']=MyGrouping_1_id_i
        

        Check=0
        while Check==0:
            MyGrouping_1_id_i+=1
            id=TestDF['PreviousXD']==Name;id
            TestDF.loc[id,'MyGrouping_1']=MyGrouping_1_i;TestDF
            TestDF.loc[id,'MyGrouping_1_id']=MyGrouping_1_id_i
            try: 
                Name=TestDF[id]['XDSegID'].iloc[0];Name
            except: Check=1
            
    DF_grouped=TestDF[['XDSegID','Miles','MyGrouping_1']].groupby('MyGrouping_1').agg({'XDSegID': ['count'], 'Miles': ['sum']}) 
    DF_grouped.columns=['Size','Miles']
    DF_grouped=DF_grouped.reset_index()
    DF_grouped=DF_grouped.rename(columns={'MyGrouping_1':'Grouping'})
    DF_grouped['Members']=  DF_grouped.apply(lambda ROW: TestDF[TestDF['MyGrouping_1']==ROW['Grouping']]['XDSegID'].tolist(),axis=1)     
    return DF_grouped, TestDF    
    





#%%
#Type 3: using XDSegID, PreviousXD, NextXDSegI and a cap for the maximum length and other thresholds
def ChangeChecker(Base1,Base2, LaneChange=False, SpeedLmtChange=False, AADTChange=False, FRC_tag=False, SlipRoad_tag=False ):
    """
    This function just check the conditions for MyGrouping_type_3 
    @param Base1: the first segment
    @param Base1: the second segment
    @param LaneChange: maximum allowable number of the lanes change for the two back to back segment in the group. False means not considering it.
    @param SpeedLmtChange: maximum allowable speed limit change for the two back to back segment in the group. False means not considering it.
    @param AADTChange: maximum allowable AADT change for the two back to back segment in the group. False means not considering it.
    @param FRC_tag: maximum allowable FRC change for the two back to back segment in the group. False means not considering it.
    @param SlipRoad_tag: maximum allowable slip road change for the two back to back segment in the group. False means not considering it.
    @return: Connected:   if the two segments satisfies the connection condition or not
    """
    
    
    Connected=True
    if LaneChange!=False:
        if abs(Base1['Lanes'].iloc[0]-Base2['Lanes'].iloc[0])>=LaneChange:
            Connected=False
    if SpeedLmtChange!=False:
        if abs(Base1['SPD_LMT'].iloc[0]-Base2['SPD_LMT'].iloc[0])>=SpeedLmtChange:
            Connected=False
    if AADTChange!=False:
        if  abs(Base1['AADT'].iloc[0]-Base2['AADT'].iloc[0])/abs(Base1['AADT'].iloc[0])>=AADTChange:
            Connected=False
    if FRC_tag!=False:
        if abs(Base1['FRC'].iloc[0]-Base2['FRC'].iloc[0])>0:
            Connected=False
    if SlipRoad_tag!=False:
        if abs(Base1['SlipRoad'].iloc[0]-Base2['SlipRoad'].iloc[0])>0:
            Connected=False            
    #print('Connected=',Connected)        
    return (Connected)



def MyGrouping_type_3(TestDF, MileCap=5, LaneChange=False, SpeedLmtChange=False, AADTChange=False, FRC_tag=False, SlipRoad_tag=False,grouped_XDSegID_Start_Number=3000000):
    """
    it groups the segments together using grouping method 3
    In grouping method 3, we just put the segments back to back based on the XDSegID, PreviousXD, NextXDSegI, and check some limitations
    @param TestDF: the inrix dataframe which includes XDSegID, XDGroup, and Miles
    @param MileCap: maximum allowable length for the group
    @param LaneChange: maximum allowable number of the lanes change for the two back to back segment in the group. False means not considering it.
    @param SpeedLmtChange: maximum allowable speed limit change for the two back to back segment in the group. False means not considering it.
    @param AADTChange: maximum allowable AADT change for the two back to back segment in the group. False means not considering it.
    @param FRC_tag: maximum allowable FRC change for the two back to back segment in the group. False means not considering it.
    @param SlipRoad_tag: maximum allowable slip road change for the two back to back segment in the group. False means not considering it.
    @return: DF_grouped:   it is a dataframe that summerized the groups, the length in miles, the size (the total number of the segments), and the list of the members
    @return: TestDF:   it is a dataframe including the orginal TestDF dataframe and more columns regarding the the group id, the member id, and end mile of the member in the group                         
    """
    
    
    if 'MyGrouping_1' in TestDF.columns:
        print("checked: MyGrouping_1 exists.")
    else:
        raise ValueError("MyGrouping_1 doesnt exist")
    TestDF['grouped_XDSegID']=0
    TestDF['grouped_XDSegID_id']=0
    TestDF['grouped_XDSegID_id_miles']=0
    MyGrouping_3_i=grouped_XDSegID_Start_Number
    for Group in sorted(TestDF['MyGrouping_1'].unique()):
        if Group!=0:
            Mile_i=0
            MyGrouping_3_id_i=0
            MyGrouping_3_i+=1
            for i in range(1,1+int(max((TestDF[TestDF['MyGrouping_1']==Group]['MyGrouping_1_id'])))):
                id=(TestDF['MyGrouping_1']==Group) & (TestDF['MyGrouping_1_id']==i)
                Mile_i=Mile_i+TestDF[id]['Miles'].iloc[0]
                Base2=TestDF.loc[id]
                if i==1:
                    Base1=Base2
                    
                Connected=ChangeChecker(Base1,Base2, LaneChange=LaneChange, SpeedLmtChange=SpeedLmtChange, AADTChange=AADTChange )
                if (Mile_i<MileCap) & (Connected==True):
                    MyGrouping_3_id_i+=1
                else:
                    MyGrouping_3_i+=1
                    MyGrouping_3_id_i=1
                    Mile_i=TestDF[id]['Miles'].iloc[0] 
                Base1=Base2
                TestDF.loc[id,'grouped_XDSegID']=MyGrouping_3_i
                TestDF.loc[id,'grouped_XDSegID_id']=MyGrouping_3_id_i
                TestDF.loc[id,'grouped_XDSegID_id_miles']=Mile_i
                
                
    DF_grouped=TestDF[['XDSegID','Miles','grouped_XDSegID']].groupby('grouped_XDSegID').agg({'XDSegID': ['count'], 'Miles': ['sum']})
    DF_grouped.columns=['Size','Miles']
    DF_grouped=DF_grouped.reset_index()
    TestDF_grouped=TestDF[['grouped_XDSegID','FRC','County_inrix','Nearest_Weather_Station']].groupby('grouped_xdsegid').agg(lambda x:x.value_counts().index[0]).reset_index()
    DF_grouped=pd.merge(DF_grouped,TestDF_grouped, left_on='grouped_xdsegid', right_on='grouped_xdsegid', how='left' )    
    
    #DF_grouped=DF_grouped.rename(columns={'MyGrouping_3':'Grouping'})
    DF_grouped['Members']=  DF_grouped.apply(lambda ROW: TestDF[TestDF['grouped_XDSegID']==ROW['grouped_XDSegID']]['XDSegID'].tolist(),axis=1)
    DF_grouped['geometry']=DF_grouped.apply(lambda ROW: Reshaper(ROW,TestDF),axis=1)
                
    return(DF_grouped,TestDF)


    """
    it groups the segments together using grouping method 1
    In grouping method 1, we use 'XDGroup' to put the segments together. 
    @param inrix_df: the inrix dataframe which includes XDSegID, XDGroup, and Miles
    @return: DF_grouped:   it is a dataframe that summerized the groups, the length in miles, the size (the total number of the segments), and the list of the members
    @return: TestDF:   it is a dataframe including the orginal TestDF dataframe and more columns regarding the the group id and the member id                           
    """
#%%

def MyGrouping_Grid(inrix_df,grouped_XDSegID_Start_Number=8000000,width = 0.1, height = 0.1, Source_crs='EPSG:4326', Intended_crs='EPSG:4326'):
    '''
    This function groups the segments based on the griding. 

    Parameters
    ----------
    inrix_df : DataFrame
        inrix DataFrame, which contains geometry     
    grouped_XDSegID_Start_Number : float
        Just a base number for grouping id.
    width, height : float
        height and width if the grids
    Source_crs,Intended_crs: Str
        you can choose between 'EPSG:4326' and 'EPSG:3310'.

    Returns
    -------
    Adding columns regarding to the grouping based on grids to the orignal inrix DataFrame.

    '''
    #inrix_df = pd.read_pickle('data/cleaned/inrix.pkl')
    inrix_gdf_4326 = inrix_df.copy()
    inrix_gdf_4326['line']=inrix_gdf_4326['geometry']
    inrix_gdf_4326 = gpd.GeoDataFrame(inrix_gdf_4326, geometry=inrix_gdf_4326['line'], crs={'init': Source_crs}).to_crs(Intended_crs)
    inrix_gdf_4326['center']=inrix_gdf_4326.centroid
    inrix_gdf_4326 = gpd.GeoDataFrame(inrix_gdf_4326, geometry=inrix_gdf_4326['center'], crs={'init': Intended_crs})     
    

    


    
    xmin,ymin,xmax,ymax =  inrix_gdf_4326['geometry'].total_bounds
    xmin=xmin-width/20
    xmax=xmax+width/20
    ymin=ymin-height/20
    ymax=ymax+height/20
    
    
    
    
    rows = int(np.ceil((ymax-ymin) /  height))
    cols = int(np.ceil((xmax-xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax- height
    polygons = []
    X_id=[]
    Y_id=[]
    for i in range(cols):
       Ytop = YtopOrigin
       Ybottom =YbottomOrigin
       for j in range(rows):
           polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
           X_id.append(i)
           Y_id.append(j)
           Ytop = Ytop - height
           Ybottom = Ybottom - height
       XleftOrigin = XleftOrigin + width
       XrightOrigin = XrightOrigin + width   
    Grid = pd.DataFrame({'geometry':polygons,'X_id':X_id,'Y_id':Y_id }).reset_index().rename(columns={'index':'grouped_XDSegID'})
    Grid['grouped_XDSegID']=Grid['grouped_XDSegID']+grouped_XDSegID_Start_Number
    Grid = (gpd.GeoDataFrame(Grid, geometry=Grid['geometry'], crs={'init': Intended_crs} ))
    inrix_gdf_4326 = gpd.sjoin(inrix_gdf_4326[['XDSegID','geometry', 'Miles','FRC','County_inrix','Nearest_Weather_Station']],
                     Grid[['grouped_XDSegID','geometry']],
                     how="left", op='within').drop('index_right',axis=1)
    
    #Fig1,ax=plt.subplots(figsize=[20,5])
    #Fig1=Grid.plot(column=(np.log(1+Grid['Size'])-1),ax=ax,  legend = True) 
    #inrix_gdf_4326[inrix_gdf_4326['MyGrouping_3'].isna()].plot(ax=Fig1,color = 'blue',markersize=0.01)   
    
    
    
    
    
    inrix_df=pd.merge(inrix_df,inrix_gdf_4326[['XDSegID','grouped_XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left' )
    inrix_df['grouped_XDSegID_id']=np.nan
    inrix_df['grouped_XDSegID_id_miles']=np.nan
    
    Grid = gpd.sjoin(Grid[['grouped_XDSegID','geometry','X_id','Y_id']], inrix_gdf_4326[['XDSegID','Miles','geometry','County_inrix']],
                     how="left", op='contains').drop('index_right',axis=1)
    

    
    DF_grouped=Grid[['XDSegID','geometry','Miles','grouped_XDSegID','X_id','Y_id']].groupby('grouped_XDSegID').agg({'XDSegID': ['count'],'Miles': ['sum'], 'geometry': ['first'],'X_id': ['first'],'Y_id': ['first']})
    DF_grouped.columns=['Size','Miles','geometry','X_id','Y_id']
    DF_grouped=DF_grouped.reset_index()
    TestDF_grouped=Grid[['grouped_XDSegID','FRC','County_inrix','Nearest_Weather_Station']].groupby('grouped_xdsegid').agg(lambda x:x.value_counts().index[0]).reset_index()
    DF_grouped=pd.merge(DF_grouped,TestDF_grouped, left_on='grouped_xdsegid', right_on='grouped_xdsegid', how='left' )        
    DF_grouped['Members']=  DF_grouped.apply(lambda ROW: Grid[Grid['grouped_XDSegID']==ROW['grouped_XDSegID']]['XDSegID'].tolist(),axis=1)
                
    
    
    
    
    
    
    #%%Graphing
    sns.set()

    Fig,ax=plt.subplots(2,1,figsize=[20,10])
    Fig1=(gpd.GeoDataFrame(DF_grouped, geometry=DF_grouped['geometry'], crs={'init': Intended_crs} )).plot(ax=ax[0],  legend = True); ax[0].set_title('Boundary of each grid');#ax[0].set_xlim(-91, -81);ax[0].set_ylim(34.5, 37) 
    Fig2=(gpd.GeoDataFrame(inrix_df, geometry=inrix_df['geometry'], crs={'init': Intended_crs} )).plot(ax=ax[1],  legend = True); ax[1].set_title('Center of each grid');#ax[1].set_xlim(-91, -81);ax[1].set_ylim(34.5, 37)      
        
    
    
    
    return(DF_grouped,inrix_df)    
    







#%%

def GroupMaker(inrix_df,MetaData, Type_Tag='Linear' ):
    '''
    This the main function that runs the other functions to complete the grouping process
    @param inrix_df:
    @MetaData:
    @Type_Tag: Linear or Grid
    @retrun inrix_df:
    '''
    if Type_Tag=='Line':
        DF_grouped, inrix_df=MyGrouping_type_1(inrix_df)
        #Save_DF(DF_grouped, MetaData['destination']+'inrix/'+'grouped/grouped_1',Format='pkl', gpd_tag=False)
        
        DF_grouped, inrix_df=MyGrouping_type_3(inrix_df, MileCap=MetaData['Grouping']['MileCap'],
                                                         LaneChange=MetaData['Grouping']['LaneChange'], 
                                                         SpeedLmtChange=MetaData['Grouping']['SpeedLmtChange'],
                                                         AADTChange=MetaData['Grouping']['AADTChange'],
                                                         FRC_tag=MetaData['Grouping']['FRC_tag'],
                                                         SlipRoad_tag=MetaData['Grouping']['SlipRoad_tag'],
                                                         grouped_XDSegID_Start_Number=MetaData['Grouping']['grouped_XDSegID_Start_Number'])
        DF_grouped.columns = DF_grouped.columns.str.lower().str.replace(' ', '_')
        Save_DF(DF_grouped, MetaData['destination']+'inrix/'+'grouped/grouped_inrix',Format='pkl', gpd_tag=False) #DF_grouped = pd.read_pickle('data/cleaned/grouped/grouped_3.pkl')

    elif Type_Tag=='Grid':
        DF_grouped, inrix_df=MyGrouping_Grid(inrix_df,grouped_XDSegID_Start_Number=MetaData['Grouping']['grouped_XDSegID_Start_Number'], width = MetaData['Grouping']['width'], height = MetaData['Grouping']['height'],
                                             Source_crs=MetaData['Grouping']['Source_crs'], Intended_crs=MetaData['Grouping']['Intended_crs'])
        DF_grouped.columns = DF_grouped.columns.str.lower().str.replace(' ', '_')
        Save_DF(DF_grouped, MetaData['destination']+'inrix/'+'grouped/grid',Format='pkl', gpd_tag=False) #DF_grouped = pd.read_pickle('data/cleaned/grouped/grid.pkl')
    return inrix_df,DF_grouped






    






