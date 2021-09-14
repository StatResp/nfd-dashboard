# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:05:39 2021

@author: Sayyed Mohsen Vazirizade 
"""
import numpy as np
import pandas as pd
#from reporting.generate_report import generate_report
import _pickle as cPickle
import bz2
#%%

#%%
def update_meta_features(met_features, df_features, cat_col):
    """
    updates feature set with transformed names for categorical features and deletes old features
    @param met_features: features from metadata
    @param df_features: features from the transformed dataframe
    @param cat_col: names of categorical columns
    @return:
    """
    try:
        for f in cat_col:
            f_cat = []
            for i in df_features:
                if "cat_" + f + '_' in i:
                    f_cat.append(i)
            met_features.remove(f)
            met_features.extend(f_cat)
    except:
        print('No Categorical Data Found!')
    #if 'Intercept' not in met_features:    
    #    met_features.extend(['Intercept'])
    return met_features

def dummy_maker(df, cat_features):
    '''
    This function creates dummy columns for the categorical features

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    cat_features : list
        list of categorical features.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    for col in cat_features:
                # create dummy dataframe
                dummy_temp = pd.get_dummies(df[col], prefix='cat_'+col)
                # merge with original dataframe
                df = pd.concat([df, dummy_temp], axis=1)
                # drop the categorical column
                #df.drop([col], axis=1, inplace=True)    
    return df

#%%

def Results_Generator(model_type,model, Conf_Matrix, test_likelihood, test_likelihood_all, pred_likelihood, pred_likelihood_all,test_MSE, test_MSE_all, pred_MSE,pred_MSE_all):
    results = {   'test_likelihood': test_likelihood,
                  'test_likelihood_all': test_likelihood_all,
                  'predict_likelihood': pred_likelihood,
                  'predict_likelihood_all': pred_likelihood_all,
                  'test_MSE': test_MSE,
                  'test_MSE_all': test_MSE_all,
                  'predict_MSE': pred_MSE,     
                  'predict_MSE_all': pred_MSE_all, 
                  "accuracy": Conf_Matrix['accuracy'], "precision": Conf_Matrix['precision'], "recall": Conf_Matrix['recall'], "f1":Conf_Matrix['f1'],
                  "accuracy_all": Conf_Matrix['accuracy_all'], "precision_all": Conf_Matrix['precision_all'], "recall_all": Conf_Matrix['recall_all'], "f1_all":Conf_Matrix['f1_all'],
                  "threshold": Conf_Matrix['threshold'],
                  "threshold_all": Conf_Matrix['threshold_all']} 
    
    if (model_type in ['SR','LR','ZIP']):
        results['train_likelihood']= model.model_stats['train_likelihood']
        results['train_likelihood_all']= model.model_stats['train_likelihood_all']
        results['aic']= model.model_stats['aic']
        results['aic_all']= model.model_stats['aic_all']    
    else:
        results['train_likelihood']= None
        results['train_likelihood_all']= [None]
        results['aic']= None
        results['aic_all']= [None]
        
    
    
    
    return(results)




#%%


def Conf_Mat(df, metadata, current_model):
    '''
    This function calculates the confusion matrix for classification models

    Parameters
    ----------
    df : dataframe
        Includes all of our data.
    metadata : dict
        metadata.
    model_name : string
        name of the model.

    Returns
    -------
    Dic
        the calculated values for accuracy, precision, recall, and F1-score.

    '''
    '''
    Name_of_classification_Col='predicted_TF'
    df=df_test[[metadata['pred_name_Count'],metadata['pred_name_TF'],'predicted','predicted_TF','cluster_label','threshold']]

    df.sum()
    df.max()
    
    df[(df['predicted']>0.12186066301832887) & (df['cluster_label']==1)]
    sum(df_[metadata['pred_name_TF']]!=df_['predicted_TF'])
    learn_results['1']['Logistic_Regression+RUS']['model'].model_threshold
    df_test
    '''
    def A_P_R_F1(df, Name_of_classification_Col_Actual=metadata['pred_name_TF'], Name_of_classification_Col_Predicted='predicted_TF'):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        if Name_of_classification_Col_Actual in df.columns:
            accuracy = accuracy_score (df[Name_of_classification_Col_Actual], df[Name_of_classification_Col_Predicted]) # accuracy: (tp + tn) / (p + n)
            precision= precision_score(df[Name_of_classification_Col_Actual], df[Name_of_classification_Col_Predicted]) # precision tp / (tp + fp)
            recall   = recall_score   (df[Name_of_classification_Col_Actual], df[Name_of_classification_Col_Predicted]) # recall: tp / (tp + fn)
            f1       = f1_score       (df[Name_of_classification_Col_Actual], df[Name_of_classification_Col_Predicted]) # f1: 2 tp / (2 tp + fp + fn)        
            return accuracy, precision, recall, f1
        else:
            return None, None, None, None    
    


    if (current_model['model_type'] in metadata['TF_Models']) | (current_model['model_type'] in metadata['Count_Models']) | (current_model['model_type']=='Naive') :#((current_model['model_type']=='LR') | (current_model['model_type']=='ZIP')):     #if the model can generate the columns predicted and predicted_TF, use this
        clusters=sorted(df.cluster_label.unique())
        accuracy, precision, recall, f1=A_P_R_F1(df )
        threshold=df['threshold'].mean()
        if len(clusters)==1:
            accuracy_all, precision_all, recall_all, f1_all, threshold_all=[np.nan], [np.nan], [np.nan], [np.nan],  [np.nan]
            
        else:
            accuracy_all, precision_all, recall_all, f1_all, threshold_all = [], [], [], [],[]
            for temp_cluster in clusters:
                accuracy_c, precision_c, recall_c, f1_c=A_P_R_F1(df[df['cluster_label']==temp_cluster])
                accuracy_all.append(accuracy_c)
                precision_all.append(precision_c)
                recall_all.append(recall_c)
                f1_all.append(f1_c)
                threshold_all.append(df[df['cluster_label']==temp_cluster]['threshold'].mean())
                
    elif current_model['model_type']=='Naive_':  #current_model['model_type']=='Naive_'  #non classification methods
        def Calculator_accuracy(row,model_i):
            if row[metadata['pred_name_TF']]>0:
                return 1*row[model_i]
            else:
                return 1-row[model_i]    
    
        accuracy= (df.apply(lambda row: Calculator_accuracy(row,'predicted'), axis=1)/len(df)).sum()
        precision=(df['predicted']*df[metadata['pred_name_TF']]/sum(df[metadata['pred_name_TF']])).sum()
        recall=   (df['predicted']*df[metadata['pred_name_TF']]/sum(df['predicted'])).sum()
        f1=2*recall*precision/(recall+precision)
        threshold=np.nan
        accuracy_all, precision_all, recall_all, f1_all, threshold_all=[np.nan], [np.nan], [np.nan], [np.nan],  [np.nan]
       
    
    else:   #non classification methods
        Length=len(df['cluster_label'].unique())
        accuracy, precision, recall, f1, threshold=np.nan, np.nan, np.nan, np.nan, np.nan
        accuracy_all, precision_all, recall_all, f1_all, threshold_all=[np.nan]*Length, [np.nan]*Length, [np.nan]*Length, [np.nan]*Length,  [np.nan]*Length

    
    
    return {"accuracy": accuracy, "accuracy_all": accuracy_all,
            "precision": precision, "precision_all": precision_all, 
            "recall": recall, "recall_all": recall_all,
            "f1":f1, "f1_all": f1_all,
            "threshold":threshold, "threshold_all": threshold_all}






#%%
def Naive_adder(DF_train,DF_test,learn_results,Window_Number_i ,metadata): 
    '''
    current_model={};current_model['model_type']='Naive'
    DF=DF_train[['XDSegID',metadata['pred_name_TF']]].groupby('XDSegID').agg({metadata['pred_name_TF']: ['count','mean']})
    DF.columns=['count', 'predicted']
    DF['predicted_TF']=None
    DF['cluster_label']=0
    DF_test=pd.merge(DF,DF_test[['XDSegID', 'time_local',metadata['pred_name_Count'],metadata['pred_name_TF']]], left_on= 'XDSegID'  ,right_on=  'XDSegID', how='right'  )
    '''
    Conf_Matrix = Conf_Mat(DF_test, metadata, current_model=current_model)  
    learn_results[Window_Number_i]['Naive']={}
    learn_results[Window_Number_i]['Naive']['df_test']=DF_test
    learn_results[Window_Number_i]['Naive']['results']=Results_Generator('Naive',None, Conf_Matrix, None, None, None, None,None, None, None,None)
    return learn_results


#%%
def Mean_of_AllTestTime_Results(learn_results):
    #Building a DF using the metrics and different test windows
    DF_results=pd.DataFrame()
    j=0
    for Window_Number_i in learn_results.keys():
        for model_i in learn_results[Window_Number_i].keys():
            #print(model_i)
            for Parameter in learn_results[Window_Number_i][model_i].keys():
                if Parameter=='results':
                    #print(learn_results[Window_Number_i][model_i][Parameter])
                    DF_results.loc[j,'rolling_window_group']=Window_Number_i
                    DF_results.loc[j,'model']=model_i
                    for Metric in learn_results[Window_Number_i][model_i][Parameter].keys():
                        if isinstance(learn_results[Window_Number_i][model_i][Parameter][Metric], list):
                            LIST=[i for i in learn_results[Window_Number_i][model_i][Parameter][Metric]]
                            for i,_ in enumerate(LIST):
                                #print(i)
                                DF_results.loc[j,(Metric+'_'+str(i+1))] = LIST[i]
                        else:
                            #LIST=np.array(learn_results[Window_Number_i][model_i][Parameter][Metric])
                            DF_results.loc[j,Metric]=learn_results[Window_Number_i][model_i][Parameter][Metric]
                        #DF_results.loc[j,Metric]=LIST
                            

                    
                    j=j+1
    #Adding the mean values of the metrics of different test windows               
    for i in  DF_results['model'].unique():
            DF_results.loc[j,:]=DF_results[DF_results['model']==i].mean()
            DF_results.loc[j,'rolling_window_group']='Mean'
            DF_results.loc[j,'model']=i
            j=j+1  
    return  DF_results

       
def Concat_AllTestTime(learn_results,metadata,Type='df_test'):    
    '''
    It includes all of the predicted data.  If rolling window is selected for training and test, the prediction for different rollowing windows are appeneded to each other. 

    Parameters
    ----------
    learn_results : TYPE
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.
    Type : TYPE, optional
        DESCRIPTION. The default is 'df_test'.

    Returns
    -------
    DF : TYPE
        DESCRIPTION.

    '''
        
    DF=pd.DataFrame()
    #Briginging all of the different rolling windows in one dataframe
    for Window_Number_i in learn_results.keys():
        DF_Temporary=pd.DataFrame()
        for model_i  in learn_results[Window_Number_i].keys():
            DF_Temporary[model_i]=learn_results[Window_Number_i][model_i][Type]['predicted']
            DF_Temporary[model_i+'_TF']=learn_results[Window_Number_i][model_i][Type]['predicted_TF']
            
        try:
            DF_Temporary[[metadata['unit_name'],'time_local',metadata['pred_name_Count'],metadata['pred_name_TF']]] =learn_results[Window_Number_i][model_i][Type][[metadata['unit_name'],'time_local',metadata['pred_name_Count'],metadata['pred_name_TF']]] 
        except:
            DF_Temporary[[metadata['unit_name'],'time_local']] =learn_results[Window_Number_i][model_i][Type][[metadata['unit_name'],'time_local']] 
            
            
        DF_Temporary['rolling_window_group']=Window_Number_i
        DF=DF.append(DF_Temporary) 

    #Just reordering the columns
    Colunms=DF.columns.copy()  
    try:
        BegList=[metadata['unit_name'],'time_local','rolling_window_group',metadata['pred_name_Count'],metadata['pred_name_TF']]
        BegList.extend(Colunms[Colunms.isin(BegList)==False].tolist())
        DF=DF[BegList]
    except:
        BegList=[metadata['unit_name'],'time_local','rolling_window_group']
        BegList.extend(Colunms[Colunms.isin(BegList)==False].tolist())
        DF=DF[BegList]        
       
    return  DF   




def Add_to_Dic(learn_results,DF_results):
    #Saving the mean values as a dictionary        
    learn_results['Mean']={}  
    for i in  DF_results['model'].unique():
        learn_results['Mean'][i]={} 
        learn_results['Mean'][i]['results']={}
        Columns=pd.Series(DF_results[(DF_results['rolling_window_group']=='Mean') & (DF_results['model']==i)].columns)
        Window_Number_i=list(learn_results.keys())[0]
        Columns_search=list(learn_results[Window_Number_i][i]['results'].keys())
        for j in Columns_search:
            if Columns.isin([j]).sum()==1:
                learn_results['Mean'][i]['results'][j]=DF_results[(DF_results['rolling_window_group']=='Mean') & (DF_results['model']==i)][j].iloc[0]
            else:
                search_str = '^' + j 
                learn_results['Mean'][i]['results'][j]=   (DF_results[(DF_results['rolling_window_group']=='Mean') & (DF_results['model']==i)][Columns[Columns.str.contains(search_str)].tolist()]).iloc[0].tolist()
    
      
    
    for Window_Number_i in  learn_results.keys():
       for model_i in  learn_results[Window_Number_i].keys(): 
           learn_results[Window_Number_i][model_i]['results']['spearman_corr']=DF_results[(DF_results['rolling_window_group']==Window_Number_i) & (DF_results['model']==model_i)]['spearman_corr'].iloc[0]
           learn_results[Window_Number_i][model_i]['results']['pearson_corr']=DF_results[(DF_results['rolling_window_group']==Window_Number_i) & (DF_results['model']==model_i)]['pearson_corr'].iloc[0]
        
    return learn_results


A=pd.DataFrame({'a':[1,2,3],'b':[None, None, 1]})
A.corr()

def Correlation_caluclator(DF, Columns_List,metadata,Type='spearman'):

    
    Time_aggregation=['rolling_window_group']
    #Time_aggregation=['week']
    Corr_Matrix=pd.DataFrame()   
    Columns_List=Columns_List
    try: 
        DF_margin_Space=DF[Time_aggregation+[metadata['unit_name']]+Columns_List].groupby(Time_aggregation+[metadata['unit_name']]).mean()
    except:
        Columns_List=Columns_List[0:-1]
        DF_margin_Space=DF[Time_aggregation+[metadata['unit_name']]+Columns_List].groupby(Time_aggregation+[metadata['unit_name']]).mean()
    for window_i in DF[Time_aggregation[0]].unique():     
        try:        
            DF_Temporary=(DF_margin_Space.loc[(slice(window_i,window_i), slice(None)), :].corr(method = Type).loc[[metadata['pred_name_TF']]])
            DF_Temporary=DF_Temporary.drop(metadata['pred_name_TF'], axis=1)
        except:
            DF_Temporary=pd.DataFrame(index=[metadata['pred_name_TF']],columns=Columns_List )
        DF_Temporary['Time_aggregation']=window_i
        Corr_Matrix=Corr_Matrix.append(DF_Temporary)
      
        
    Corr_Matrix.loc['Mean'] = Corr_Matrix.mean()    
    Corr_Matrix.loc['Mean','Time_aggregation']='Mean'
    Corr_Matrix=Corr_Matrix.reset_index().drop('index',axis=1)

    return Corr_Matrix

def Correctness_caluclator(DF_Test_spacetime,DF_Test_metric_time,DF_results, learn_results,metadata, Type=1 ):
    if Type==1:
        Name='Correctness'
    else:
        Name='Correctness'+str(Type)
    if metadata['pred_name_TF'] in DF_Test_spacetime.columns:
        #Time_aggregation=['rolling_window_group']
        #window_i = DF[Time_aggregation[0]].unique()
        #DF=DF_Test_spacetime[DF_Test_spacetime[Time_aggregation[0]]==window_i[0]].copy()
        Count=DF_Test_spacetime.groupby(['time_local'])[metadata['pred_name_TF']].sum().reset_index()
        Count=Count.rename(columns={metadata['pred_name_TF']:'Count'})
        Total_len=round(len(DF_Test_spacetime[metadata['unit_name']].unique())*Type)
        
        Window_Number_i=list(learn_results.keys())[0]   
        for model_i in  learn_results[Window_Number_i].keys():
            Count_Succes=[]
            for i in range(0,len(Count)):
                if Count.iloc[i]['Count']==0:
                    Count_Succes.append(np.nan)
                else:
                    if isinstance(Type,int):
                        Count_Succes.append(DF_Test_spacetime[DF_Test_spacetime['time_local']==Count.iloc[i]['time_local']].sort_values(model_i,ascending=False)[0:Type*Count.iloc[i]['Count']][metadata['pred_name_TF']].sum()/Count.iloc[i]['Count'])
                    elif isinstance(Type,float):
                        Count_Succes.append(DF_Test_spacetime[DF_Test_spacetime['time_local']==Count.iloc[i]['time_local']].sort_values(model_i,ascending=False)[0:Total_len][metadata['pred_name_TF']].sum()/Count.iloc[i]['Count'])
            Count[model_i+'_'+Name]=Count_Succes
        Count=Count.drop('Count', axis=1)
    else:
        Window_Number_i=list(learn_results.keys())[0]   
        Count=DF_Test_spacetime.groupby(['time_local'])[metadata['unit_name']].count().reset_index().rename(columns={metadata['unit_name']:'Count'})
        for model_i in  learn_results[Window_Number_i].keys():        
            Count[model_i+'_'+Name]=None
            #Count=Count.drop('Count', axis=1)
    
    
    DF_Test_metric_time=pd.merge(DF_Test_metric_time,Count, left_on='time_local', right_on='time_local', how='left' )
    if 'Count' in DF_Test_metric_time.columns:
        DF_Test_metric_time=DF_Test_metric_time.drop('Count', axis=1)
    Count=pd.merge(Count,DF_Test_metric_time[['time_local','rolling_window_group']],left_on='time_local',right_on='time_local', how='left')
    A=Count.groupby('rolling_window_group',dropna=False).mean()
    for i in Count.columns:
        if (not i in A.columns) and (i!='rolling_window_group'):
            A[i]=None
    if 'Count' in A.columns:
        A=A.drop('Count', axis=1)
    A.loc['Mean']=A.mean()
    A=A.reset_index()
    A=pd.melt(A, id_vars=['rolling_window_group']).rename(columns={'rolling_window_group':'rolling_window_group','variable':'model','value':Name })
    A['model']=A['model'].str.replace('_'+Name,'')
    
    
    DF_results=pd.merge(DF_results,A, left_on=['rolling_window_group','model'], right_on=['rolling_window_group','model'], how='left' )
    return DF_Test_metric_time,DF_results


def Correlation_Function(DF,DF_results,metadata):
    
    Columns_List=list(np.append(DF_results['model'].unique(),[metadata['pred_name_TF']]))
    #Columns_List=list(DF_results['model'].unique())
    Corr_Matrix_spearman=Correlation_caluclator(DF,Columns_List,metadata,Type='spearman')
    Corr_Matrix_pearson=Correlation_caluclator(DF,Columns_List,metadata,Type='pearson')
    
    DT1=pd.melt(Corr_Matrix_spearman, id_vars=['Time_aggregation'], value_vars=Columns_List[:-1])
    DT1=DT1.rename(columns={'value':'spearman_corr'})
    DT2=pd.melt(Corr_Matrix_pearson, id_vars=['Time_aggregation'], value_vars=Columns_List[:-1])
    DT2=DT2.rename(columns={'value':'pearson_corr'})
    DT=pd.merge(DT1,DT2,left_on=['Time_aggregation','variable'],right_on=['Time_aggregation','variable'], how='inner' )
    DT=DT.rename(columns={'variable':'model','Time_aggregation':'rolling_window_group'})
    DF_results=pd.merge(DF_results,DT,left_on=['rolling_window_group','model'],right_on=['rolling_window_group','model'], how='left' )
    
    
    #DF_results_[(DF_results_['rolling_window_group']==Window_Number_i) & (DF_results_['model']==model_i)]['spearman_corr'].iloc[0]

    return DF_results



   

    

def Metric_calculator_per_time(DF_Test_spacetime, DF_results,metadata):
    '''
    This function generates the figure that shows the accuracy, precision, reall, and F1-score using the naive model and the prediction models for 1 month

    Parameters
    ----------
    DF_Test_spacetime : DF
        DESCRIPTION.
    DF_results : DF
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    from statresp.plotting.figuring_prediction import Graph_Metric
    def Calculator_accuracy(row,model_i):
        if metadata['pred_name_TF'] in row.index:
            if row[metadata['pred_name_TF']]>0:
                return 1*row[model_i]
            else:
                return 1-row[model_i]
        else:
            return None
    '''
    DF_Temporary=pd.DataFrame()
    DF_Temporary['time_local']=DF_Test_spacetime['time_local']
    for time_local_i in DF_Test_spacetime['time_local'].drop_duplicates().sort_values():
        Mask=DF_Test_spacetime['time_local']==time_local_i
        for model_i  in (DF_results['model'].unique()): 
            DF_Temporary.loc[Mask,model_i+'_accuracy']=DF_Test_spacetime[Mask].apply(lambda row: Calculator_accuracy(row,model_i), axis=1)/len(DF_Test_spacetime[Mask])
            DF_Temporary.loc[Mask,model_i+'_recall']=DF_Test_spacetime[Mask][model_i]*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/((DF_Test_spacetime[Mask][metadata['pred_name_TF']]).sum())
            DF_Temporary.loc[Mask,model_i+'_precsion']=DF_Test_spacetime[Mask][model_i]*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/((DF_Test_spacetime[Mask][model_i+'_TF']).sum()) 
        DF_Temporary.loc[Mask,'Naive_accuracy']=DF_Test_spacetime[Mask].apply(lambda row: Calculator_accuracy(row,'Naive'), axis=1)/len(DF_Test_spacetime[Mask])
        DF_Temporary.loc[Mask,'Naive_recall']=DF_Test_spacetime[Mask]['Naive']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/sum(DF_Test_spacetime[Mask][metadata['pred_name_TF']])
        DF_Temporary.loc[Mask,'Naive_precsion']=DF_Test_spacetime[Mask]['Naive']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/sum(DF_Test_spacetime[Mask]['Naive'])
    DF_time=DF_Temporary.groupby('time_local').sum().reset_index()
    for model_i  in DF_results['model'].unique():
        DF_time[model_i+'_F1']=2*DF_time[model_i+'_recall']*DF_time[model_i+'_precsion']/(DF_time[model_i+'_recall']+DF_time[model_i+'_precsion'])
        DF_time[model_i+'_F1'].fillna(0, inplace = True) 
    DF_time['Naive_F1']=2*DF_time['Naive_recall']*DF_time['Naive_precsion']/(DF_time['Naive_recall']+DF_time['Naive_precsion'])
    DF_time['Naive_F1'].fillna(0, inplace = True) 
    DF_time.mean()    
    if metadata['figure_tag']==True:
        Graph_Metric(DF_time,'Comparing Total_Number_Incidents')
    '''
    
    
    
    DF_Temporary=pd.DataFrame()
    DF_Temporary['time_local']=DF_Test_spacetime['time_local']
    for time_local_i in DF_Test_spacetime['time_local'].drop_duplicates().sort_values():
        Mask=DF_Test_spacetime['time_local']==time_local_i
        for model_i  in (DF_results['model'].unique()): 
            if metadata['pred_name_TF'] in DF_Test_spacetime.columns:
                if model_i!='Naive':
                    DF_Temporary.loc[Mask,model_i+'_accuracy']=DF_Test_spacetime[Mask].apply(lambda row: Calculator_accuracy(row,model_i+'_TF'), axis=1)/len(DF_Test_spacetime[Mask])
                    DF_Temporary.loc[Mask,model_i+'_recall']=DF_Test_spacetime[Mask][model_i+'_TF']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/((DF_Test_spacetime[Mask][metadata['pred_name_TF']]).sum())
                    DF_Temporary.loc[Mask,model_i+'_precsion']=DF_Test_spacetime[Mask][model_i+'_TF']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/((DF_Test_spacetime[Mask][model_i+'_TF']).sum()) 
                elif model_i=='Naive':    
                    DF_Temporary.loc[Mask,'Naive_accuracy']=DF_Test_spacetime[Mask].apply(lambda row: Calculator_accuracy(row,'Naive'), axis=1)/len(DF_Test_spacetime[Mask])
                    DF_Temporary.loc[Mask,'Naive_recall']=DF_Test_spacetime[Mask]['Naive']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/sum(DF_Test_spacetime[Mask][metadata['pred_name_TF']])
                    DF_Temporary.loc[Mask,'Naive_precsion']=DF_Test_spacetime[Mask]['Naive']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/sum(DF_Test_spacetime[Mask]['Naive'])
            else:
                if model_i!='Naive':
                    DF_Temporary.loc[Mask,model_i+'_accuracy']=None
                    DF_Temporary.loc[Mask,model_i+'_recall']=None
                    DF_Temporary.loc[Mask,model_i+'_precsion']=None 
                elif model_i=='Naive':    
                    DF_Temporary.loc[Mask,'Naive_accuracy']=None 
                    DF_Temporary.loc[Mask,'Naive_recall']=None 
                    DF_Temporary.loc[Mask,'Naive_precsion']=None
                


    DF_time_TF=DF_Temporary.groupby('time_local').sum().reset_index()
    for model_i  in DF_results['model'].unique():
        DF_time_TF[model_i+'_F1']=2*DF_time_TF[model_i+'_recall']*DF_time_TF[model_i+'_precsion']/(DF_time_TF[model_i+'_recall']+DF_time_TF[model_i+'_precsion'])
        DF_time_TF[model_i+'_F1'].fillna(0, inplace = True) 
    #DF_time_TF['Naive_F1']=2*DF_time_TF['Naive_recall']*DF_time_TF['Naive_precsion']/(DF_time_TF['Naive_recall']+DF_time_TF['Naive_precsion'])
    #DF_time_TF['Naive_F1'].fillna(0, inplace = True) 
    #DF_time_TF.mean()
    
    #if metadata['figure_tag']==True:
    #    Graph_Metric(DF_time_TF,'Comparing Total_Number_Incidents_TF, Main Figure')
    
    DF_time_TF=pd.merge(DF_time_TF,DF_Test_spacetime[['time_local','rolling_window_group']].drop_duplicates(),left_on='time_local',right_on='time_local', how='left')
    
    return DF_time_TF


def Table_Generator(Window_Number_i,learn_results,metadata,Address=None):
    '''
    This fuction summerizes all of the resutls in a dictionary called  results[model_i]

    Parameters
    ----------
    Window_Number_i : str
        We may have multiple test windows. This parameter defines which Test window we want to make the table for.
    learn_results : Dict
        This dictionary contains all of the results. 
    metadata : Dict
        DESCRIPTION.
    Type : String, optional
        It is usually either df_test or df_predict based on what you want to draw. The default is 'df_test'.
    Address : TYPE, str
        Where you want the figures to be saved. The default is None.

    Returns
    -------
    None.

    '''        
    results={}
    for m in  learn_results[Window_Number_i]:   #This for loop goes over all of the models in the Window_Number_i
        model_i   = learn_results[Window_Number_i][m]['model'].metadata['Name']       #it is the name of the model: such as LR+RUS1.0+KM2, LR
        results[model_i]=learn_results[Window_Number_i][m]['results']   
    #generate_report(results,metadata['cluster_number']+[''],Address+'Report_'+metadata['model_type'][0]+'_'+'_testwindow('+Window_Number_i+')'+'.html' )   




def SQL_code_generator(Segment_List,Address_to_Save='D:/inrix_new/inrix_pipeline/else/sql.txt'):
    #'D:/inrix_new/inrix_pipeline/matrix_for_Geof/sql.txt'
    '''
    sql
    SELECT xd_id, 
           MIN(DATE_FORMAT(measurement_tstamp,'%Y-%m-%d %H:%i:%s')) AS measurement_tstamp,
           AVG(average_speed) AS average_speed_mean,
           EXTRACT (HOUR FROM measurement_tstamp) / 4 AS window
    FROM "trafficdb"."speeds"
    WHERE year IN (2019)
        AND month IN (1)
        AND countyid=1
        AND DATE(measurement_tstamp) >= DATE('2019-01-01')
        AND DATE(measurement_tstamp) <= DATE('2019-01-05')
        AND xd_id IN (156185911, 1524646417)
    GROUP BY xd_id,
             EXTRACT (HOUR FROM measurement_tstamp) / 4   
    order by 2
    limit 100 ;
       
    '''
    
    
    
    #Python to sql
    List_segments=Segment_List
    #List_segments=[156185911, 1524646417]
    print('All list of the segments:', len(List_segments))
    
    Querry=['SELECT xd_id, '
            'extract(HOUR FROM measurement_tstamp)/',str(4),' AS window, '
            'min(DATE_FORMAT(measurement_tstamp,\'%Y-%m-%d %H:%i:%s\')) as measurement_tstamp, '
            'avg(travel_time_seconds) as travel_time_seconds_mean, '
            'avg(reference_speed) as reference_speed_mean '
            'FROM "trafficdb"."speeds" ',
            'WHERE '
            'year in (2020)  AND month in (2) AND countyid in (4,1,5,12,7,17,66,10) AND '
            'DATE(measurement_tstamp) >= ', 'DATE(\'', '2020-02-01', '\') ',
            'and DATE(measurement_tstamp) <= ', 'DATE(\'', '2020-02-28', '\') and ',
            'xd_id in ', str(tuple(List_segments)), ' ',
                'group by xd_id, '
                'extract(HOUR FROM measurement_tstamp)/',str(4)
                ]
    
    Querry_str=''
    for i in Querry:
        Querry_str=Querry_str+i
    #print(Querry_str.replace("\'","'")+' limit 10')
    file1 = open(Address_to_Save,"w")
    file1.writelines(Querry_str.replace("\'","'"))
    file1.close() 
    












#def predict(model, df, metadata):
#    #smv: not needed anymore
#    """
#    Wrapper method before data is passed to specific predict methods for regression models
#    @param model: the trained model
#    @param df: dataframe of points where predictions need to be made
#    @param metadata: dictionary with start and end dates for predictions, cluster labels etc
#    @return: dataframe with E(y|x) appended to each row
#    """
#    if model.name == 'Poisson_Regression' or model.name == 'Negative_Binomial_Regression' or model.name == 'Simple_Regression':
#        #df_ = create_regression_df_test(df, metadata)
#        df_=df
#        df_samples = model.predict(df_, metadata)
#        return df_samples