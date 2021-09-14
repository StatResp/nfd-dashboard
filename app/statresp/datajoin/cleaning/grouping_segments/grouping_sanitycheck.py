# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:38:25 2020

@author: vaziris
"""
#%%

from pprint import pprint
import pandas as pd
import geopandas as gpd 
import numpy as np 


#Sanity Check: this sanity check may take a long time to be processed!!!!!
#This sanity check is specilized for inrix data.

TestDF=inrix_df
for ID in range(len(TestDF)):
    #print(ID)
    SUM=sum(TestDF['NextXDSegI']== TestDF.iloc[ID]['XDSegID'])
    if SUM>1: print(ID,'NextXDSegI is repeated twice')
    #else: print('No XDSegID is repeated at least twice as NextXDSegI')
    
    SUM=sum(TestDF['PreviousXD']== TestDF.iloc[ID]['XDSegID'])
    if SUM>1: print(ID,'PreviousXD is repeated twice')  
    #else: print('No XDSegID is repeated at least twice as PreviousXD')

    SUM=sum((TestDF['PreviousXD']== TestDF.iloc[ID]['XDSegID']) & (TestDF['NextXDSegI']== TestDF.iloc[ID]['XDSegID']))
    if SUM>0: print(ID,'PreviousXD and NextXDSegI are the same')  
    #else: print('No XDSegID is repeated for another segemnt as both PreviousXD and NextXDSegI ')    
    
    if not np.isnan(TestDF.iloc[ID]['NextXDSegI']):
        SUM=sum(TestDF['XDSegID']== TestDF.iloc[ID]['NextXDSegI'])
        if SUM==0: print(ID,'NextXDSegI exists but doesnt exist in XDSegID')
        #else: print('No NextXDSegI is unrepeated in XDSegID')    
    
    if not np.isnan(TestDF.iloc[ID]['PreviousXD']):
        SUM=sum(TestDF['XDSegID']== TestDF.iloc[ID]['PreviousXD'])
        if SUM==0: print(ID,'PreviousXD exists but doesnt exist in XDSegID')
        #else: print('No PreviousXD is unrepeated in XDSegID')     
    
  
    A=(TestDF[TestDF['NextXDSegI']== TestDF.iloc[ID]['XDSegID']]['XDSegID'])==TestDF.iloc[ID]['PreviousXD']
    if A.empty: 1#print('')
    elif A.iloc[0]==False: print(ID, 'They are not reciprocal')
    #else:    print('IF A is previous for B, B is next for A')    
    
    A=(TestDF[TestDF['PreviousXD']== TestDF.iloc[ID]['XDSegID']]['XDSegID'])==TestDF.iloc[ID]['NextXDSegI']
    if A.empty: 1#print('')
    elif A.iloc[0]==False: print(ID, 'They are not reciprocal')
    #else:    print('IF A is next for B, B is previous for A')  
    

        

for ID in range(len(TestDF)):
    A=(TestDF[TestDF['NextXDSegI']== TestDF.iloc[ID]['XDSegID']]['SlipRoad'])!=TestDF.iloc[ID]['SlipRoad']
    if A.empty: 1#print('')
    elif A.iloc[0]==True: 
        print(ID)
        print(TestDF[TestDF['NextXDSegI']== TestDF.iloc[ID]['XDSegID']].iloc[0])   
        print(TestDF.iloc[ID])
        print('The seg is: ',TestDF.iloc[ID]['XDSegID'])
        #input("Press Enter to continue...")

