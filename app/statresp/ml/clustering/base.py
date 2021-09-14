"""
@Author - Ayan Mukhopadhyay
Base Class for Spatial-Temporal Clustering. All clustering child classes must implement -

1. similarity_measure - a definition of the distance used to check difference between two units. typically a norm
2. setup_learning_model - for learning dependent clustering methods, setup a regression model
3. merge - merge existing clusters
4. fit - fit clusters to given data
"""
from statresp.ml.clustering.simple_clustering import Simple_Cluster
from statresp.ml.clustering.Agglomerative_hierarchical import AH_Cluster
import pandas as pd

class Clustering_Algo():

    def __init__(self, name,cluster_number):
        self.name = name
        self.n = cluster_number
        self.Cluster = None
        
        if self.name=="KM":
            self.Cluster=Simple_Cluster('KM',cluster_number)        
        elif self.name=="AH":
            self.Cluster=AH_Cluster('AH',cluster_number)   
    def similarity_measure(self, x_i, x_j, norm=None):
        pass

    def setup_learning_model(self):
        pass

    def merge(self, df, df_clusters, label_field):
        pass



    def fit(self, df_train, metadata=None):
        #fit clustering model to data
        #@param df_train: dataframe which used to count the number of accidents for each segment and use it for clustering training
        #@param df_test and df_predict: dataframe we want to apply clustering on it 
        #@param metadata: 
        #@return: a list indicates the cluster id

        if metadata['incident_pickle_address'][-4:] == '.pkl':
            df_incident = pd.read_pickle(metadata['incident_pickle_address'])
        else:
            df_incident = pd.read_parquet(metadata['incident_pickle_address'])
            from shapely import wkt
            df_incident['geometry'] = df_incident['geometry_string'].apply(wkt.loads)

        df_incident = df_incident[(df_incident[metadata['time_column']] >= metadata['start_time_train']) & (df_incident[metadata['time_column']] < metadata['end_time_train'])]

        if metadata['inrix_pickle_address'][-4:] == '.pkl':
            df_inrix = pd.read_pickle(metadata['inrix_pickle_address'])
        else:
            df_inrix = pd.read_parquet(metadata['inrix_pickle_address'])
            uncompatible_list = ['beg', 'end', 'center', 'geometry', 'geometry3d', 'geometry_highres']
            for i in uncompatible_list:
                if i in df_inrix.columns:
                    # df[i] = df[i].astype(str)
                    from shapely import wkt
                    df_inrix[i] = df_inrix[i].apply(wkt.loads)
        if not metadata['unit_name'] in df_incident.columns:
            if 'grouped_type3' in df_incident.columns:
                df_incident=df_incident.rename(columns={'grouped_type3':metadata['unit_name']})
        df_segmentrate = pd.merge(df_inrix[[metadata['unit_name']]],df_incident[['incident_id', metadata['unit_name']]],
                                  left_on =metadata['unit_name'], right_on=metadata['unit_name'], how='left' )
        df_segmentrate = df_segmentrate[['incident_id',metadata['unit_name']]].groupby(metadata['unit_name']).count().reset_index().rename(columns={'incident_id':metadata['pred_name_TF']})
        #print('len of df_segmentrate', len(df_segmentrate))
        self.Cluster.fit(df_segmentrate, metadata)

    
        
    def pred(self,df,metadata):
        if 'cluster_label' in df.columns:  
            df=df.drop('cluster_label',axis=1)               
 
        df= self.Cluster.pred(df)
        return df
 


        
