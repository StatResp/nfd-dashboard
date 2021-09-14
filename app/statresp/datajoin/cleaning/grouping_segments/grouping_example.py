# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:32:05 2020

@author: vaziris
"""

from pprint import pprint
import pandas as pd
import geopandas as gpd 
import numpy as np 



from grouping_segments.Grouping_Functions import MyGrouping_type_1, MyGrouping_type_3


#%%
TestDF={'XDSegID': [99,100,101,102, 103, 104, 105, 106, 107, 108, 109, 110],
     'PreviousXD': [np.nan,np.nan, 100,101, np.nan, 103,104,105,106,np.nan,108,109],
     'NextXDSegI': [np.nan,101,102, np.nan,104,105,106,107,np.nan, 108,109,110],
     'Miles': [1, 1,1.5,2,2.5, 3, 1,1.5,2, 2.5,3,3]     }
TestDF=pd.DataFrame(TestDF)
print(TestDF)


#Tutorial Example: MyGrouping_type_1
DF_grouped, TestDF=MyGrouping_type_1(TestDF)
print('Groups dataframe:')
print(DF_grouped)
print('Original dataframe with some columns regarding grouping:')
print(TestDF)



#Tutorial Example: MyGrouping_type_3
DF_grouped, TestDF=MyGrouping_type_3(TestDF, MileCap=3.5, LaneChange=False, SpeedLmtChange=False, AADTChange=False, FRC_tag=False, SlipRoad_tag=False)
#Applyting MyGrouping_type_1 on the FRC0 
print('Groups dataframe:')
print(DF_grouped)
print('Original dataframe with some columns regarding grouping:')
print(TestDF)