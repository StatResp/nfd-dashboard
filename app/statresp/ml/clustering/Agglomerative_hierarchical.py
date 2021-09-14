"""
@Author - Sayyed Mohsen Vazirizade
Hierarchical Clustering Model -- Iteratively checks for similarty in the
feature space and merges clusters up to a pre-specified number of clusters
"""
import pickle
import time

import pandas as pd
#from forecasters import poisson_reg_forecaster
import numpy as np
import networkx as nx
import geopandas as gpd
import scipy
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
import shapely

class AH_Cluster:
    def __init__(self,model_name,n_clusters):
        self.model_name = model_name
        self.n_clusters = n_clusters
        #self.name = "Simple Clustering"
        if (self.model_name == 'AH'):
            model = np.nan
        else:
            model=np.nan

        self.modelparam=model

        ######################
        ## Constants
        self.EUCLIDEAN_DISTANCE_THRESHOLD = 5 #km
        self.GRAPH_DISTANCE_THRESHOLD = 10
        self.RATE_DIFF_THRESHOLD = 2
        self.CLUSTERING_AFFINITY = 'euclidean'
        self.CLUSTERING_LINKAGE = 'ward'

    def fit(self, df_segmentrate, metadata=None):

        '''
        Fit the clustering model to the df_segmentrate data. Currently assumes that only frc==0 should be clustered,
        but uses all segments (regardless of frc) when determining graph connectivity. Current implemenation assumes
        df_segmentrate includes number of incidents and either unit_id or grouped unit_id columns.
        @param df_segmentrate:
        @param metadata:
        @return:
        '''

        if (self.model_name == 'AH'):

            _cluster_data_df, graph, adj_df = self.clustering_data_prep(metadata=metadata,
                                                                        df_segmentrate=df_segmentrate,
                                                                        unit_name='xdsegid')

            _cluster_data_df = self.run_agglomerative_clustering(data_df=_cluster_data_df,
                                                                 unit_name='xdsegid',
                                                                 number_of_clusters=self.n_clusters,
                                                                 features_to_cluster_on=[self.pred_name_TF],
                                                                 graph=graph,
                                                                 incident_rate_name=self.pred_name_TF,
                                                                 df_edges=adj_df)

            print('completed clustering')

            if self.unit_name == 'xdsegid':
                self.Seg_DF=_cluster_data_df[['xdsegid','cluster_label']].copy()
            else: # if full analysis is using grouped segments, regroup the clustered segment data
                self.Seg_DF=_cluster_data_df[['cluster_label',self.unit_name]].groupby([self.unit_name]).agg(lambda x:x.value_counts().index[0]).reset_index()

            self.Seg_DF=self.Seg_DF[self.Seg_DF[self.unit_name].isin(metadata['all_segments'])]

        else:
            raise ValueError('Inside Agglomerative Clustering Function, but model name != \'AH\'')


    def pred(self, df):

        '''
        Return predicted cluster labels for each row in the input dataframe
        @param df:
        @return:
        '''

        Seg_DF=self.Seg_DF
        df=pd.merge(df  ,Seg_DF, left_on=self.unit_name, right_on=self.unit_name , how='left')
        Max=Seg_DF['cluster_label'].value_counts().sort_values().index[0]
        df['cluster_label']=df['cluster_label'].fillna(Max)
        df['cluster_label']=df['cluster_label'].astype(int)
        return df

    def clustering_data_prep(self,
                             metadata,
                             df_segmentrate,
                             unit_name):
        '''

        @return:
        '''
        self.unit_name = metadata['unit_name']
        self.pred_name_TF =metadata['pred_name_TF']  # Incident rate / count

        # Load and process graph and inrix data
        graph = nx.read_gpickle(metadata['graph_pickle_address'])
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
        df_inrix = df_inrix.rename(columns={'grouped_xdsegid': self.unit_name})
        df_adj=nx.to_pandas_edgelist(graph)
        cluster_data_df=pd.DataFrame({'xdsegid':list(set(df_adj['source'].tolist()+ df_adj['target'].tolist()))})

        # Ensure the dataframe to cluster is based on xdsegid, since our graph is defined using xdsegids, but
        #   the input can either be based on xdsegid or grouped_xdsegid
        if self.unit_name == 'xdsegid':
            cluster_data_df=pd.merge(df_inrix[['xdsegid','geometry']],cluster_data_df, left_on= 'xdsegid', right_on= 'xdsegid', how='right' )
            cluster_data_df=pd.merge(cluster_data_df[['xdsegid','geometry']],df_segmentrate, left_on= 'xdsegid', right_on= 'xdsegid', how='left' )
        else:
            if 'members' in df_inrix.columns:
                df_inrix=df_inrix.explode('members').rename(columns={'members':'xdsegid'})
            elif 'Members' in df_inrix.columns:
                df_inrix=df_inrix.explode('Members').rename(columns={'Members':'xdsegid'})
            elif 'xdsegid' in df_inrix.columns:
                df_inrix=df_inrix.explode('xdsegid')

            cluster_data_df=pd.merge(df_inrix[['xdsegid','geometry',self.unit_name, 'frc']],cluster_data_df[['xdsegid']], left_on= 'xdsegid', right_on= 'xdsegid',  how='right' )
            cluster_data_df=pd.merge(cluster_data_df[['xdsegid','geometry', self.unit_name, 'frc']],df_segmentrate, left_on= self.unit_name, right_on= self.unit_name, how='left' )

        cluster_data_df[self.pred_name_TF]=cluster_data_df[self.pred_name_TF].fillna(0)
        cluster_data_df['center'] = cluster_data_df.apply(lambda row: row['geometry'].centroid, axis=1)

        cluster_data_df = self.add_degree_feature(graph, cluster_data_df, unit_name=unit_name)
        cluster_data_df = cluster_data_df[cluster_data_df[unit_name].isin(list(graph.nodes()))]
        cluster_data_df = cluster_data_df.reset_index().drop('index', axis=1)
        cluster_data_df = cluster_data_df.reset_index()

        return cluster_data_df, graph, df_adj

    def run_agglomerative_clustering(self,
                                     data_df,
                                     unit_name,
                                     number_of_clusters,
                                     features_to_cluster_on,
                                     graph,
                                     incident_rate_name,
                                     df_edges):

        '''
        Runs the agglomerative clustering algorithm. Currently runs for all
        '''

        _df_data_modified = data_df.copy()

        #######################
        ## Create connectivity matricies. Some that are currently unused are commented out to save time.

        # _euclidean_connectivity_matrix = self.create_euclidean_connectivity_matrix(_df_data=_df_data_modified,
        #                                                                       threshold=self.EUCLIDEAN_DISTANCE_THRESHOLD,
        #                                                                       unit_name=unit_name)

        _graph_connectivity_matrix = self.create_graph_connectivity_matrix(_df_data=_df_data_modified,
                                                                           _graph=graph,
                                                                           threshold=self.GRAPH_DISTANCE_THRESHOLD,
                                                                           unit_name=unit_name,
                                                                           _load=False)

        # _rate_diff_connectivity_matrix = create_rate_diff_connectivity_matrix(_df_data=_df_data_modified,
        #                                                                       threshold=self.RATE_DIFF_THRESHOLD,
        #                                                                       incident_rate_name=incident_rate_name)

        #######################
        ## Filter to frc == 0 after calculating connectivity to avoid clustering frc >= 1.
        ##  Saves on memory. Filter for both the data df and the connectivity matrix
        INDEX = _df_data_modified[_df_data_modified['frc'].isin([0])].index.tolist()
        _df_data_modified=_df_data_modified.iloc[INDEX]

        _full_connectivity_matrix=_graph_connectivity_matrix
        # _full_connectivity_matrix=np.multiply(_graph_connectivity_matrix,_euclidean_connectivity_matrix)
        _full_connectivity_matrix=_full_connectivity_matrix[INDEX,:]
        _full_connectivity_matrix=_full_connectivity_matrix[:,INDEX]
        del _graph_connectivity_matrix # delete to save on memory?

        #######################
        ## Filter to only the largest connected component in both the data df and connectivity matrix
        connected_components = self.get_connected_components(_graph_connectivity_matrix=_full_connectivity_matrix)
        largest_component_index = connected_components.value_counts().index[0][0]
        INDEX_connected_components = _df_data_modified.reset_index()[connected_components['Component_Index']==largest_component_index].index.tolist()
        _df_data_modified=_df_data_modified.iloc[INDEX_connected_components]
        _full_connectivity_matrix=_full_connectivity_matrix[INDEX_connected_components,:]
        _full_connectivity_matrix=_full_connectivity_matrix[:,INDEX_connected_components]

        ######################
        ## Perform clustering
        _cluster_df = self.cluster_helper(_data_df=_df_data_modified,
                                          connectivity_matrix=_full_connectivity_matrix,
                                          n_clusters=number_of_clusters,
                                          features=features_to_cluster_on,
                                          affinity=self.CLUSTERING_AFFINITY,
                                          linkage=self.CLUSTERING_LINKAGE,
                                          unit_name=unit_name,
                                          graph=graph,
                                          incident_rate_name=incident_rate_name,
                                          df_edges=df_edges)


        return _cluster_df

    def cluster_helper(self,
                       _data_df,
                       connectivity_matrix,
                       features,
                       unit_name,
                       graph,
                       incident_rate_name,
                       df_edges,
                       linkage='ward',
                       affinity='euclidean',
                       n_clusters=20):

        '''
        ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html
        @param _data_df:
        @param connectivity_matrix:
        @param features:
        @param unit_name:
        @param graph:
        @param incident_rate_name:
        @param df_edges:
        @param linkage:
        @param affinity:
        @param n_clusters:
        @return:
        '''

        _cluster_df = _data_df.copy()

        # if only one feature to cluster on, must add dummy feature to make it 2 dimensional for sklearn
        if len(features) == 1:
            _cluster_df['Dummy'] = 0
            features.append('Dummy')

        agg_cluster_obj = AgglomerativeClustering(n_clusters=n_clusters,
                                                  affinity=affinity,
                                                  linkage=linkage,
                                                  compute_distances=True,
                                                  connectivity=connectivity_matrix)

        print('\tbegin clustering')
        start_time = time.time()

        agg_cluster_obj = agg_cluster_obj.fit(_cluster_df[features])
        _cluster_df['cluster_label']=agg_cluster_obj.fit_predict(_cluster_df[features])
        print('\tclustering completed after {} seconds'.format(time.time() - start_time))

        ##################
        # figure
        # self.plot_dendrogram(agg_cluster_obj, truncate_mode='level', linkage=linkage)
        # self.Plot(_cluster_df, df_edges=df_edges, Show_Links_Tag=False, unit_name=unit_name,
        #           incident_rate_name=incident_rate_name)
        ##################

        return _cluster_df

    def add_degree_feature(self,
                           Graph,
                           DF_inrix,
                           unit_name):
        '''
        Add degree feature to the dataframe using networkx - degree is the number of edges adjacent to a node
        '''
        Degree=pd.DataFrame.from_dict(nx.degree(Graph))
        Degree.columns=[unit_name,'Degree']
        DF_inrix=pd.merge(DF_inrix,Degree[[unit_name,'Degree']], left_on=unit_name,right_on=unit_name, how='left')
        return DF_inrix

    def create_graph_connectivity_helper(self,
                                         segment_list,
                                         graph,
                                         data_df,
                                         method='simple',
                                         path_length_threshold=100,
                                         graph_dist_threshold=10,
                                         weight_feature='Not_Defined'):
        """
        Returns the (graph) distance between each node and every other (connected) node. Note that
        graph is directed
        """
        if method== 'simple':
            start_time = time.time()
            num_segs = len(segment_list)
            distance_matrix = np.zeros(shape=(num_segs, num_segs), dtype=np.int32)#, dtype=np.int32)
            full_df = data_df[['xdsegid']]
            for i, Seg in enumerate(segment_list):
                # graph_distance_dict[Seg] = dict(nx.single_source_shortest_path_length(graph, Seg, threshold))
                seg_graph_dist = dict(nx.single_source_shortest_path_length(graph, Seg, path_length_threshold))
                row_df = pd.DataFrame({'xdsegid': list(seg_graph_dist.keys()), 'dist':list(seg_graph_dist.values())})
                df_xdsegid=pd.merge(full_df,row_df, left_on= 'xdsegid', right_on= 'xdsegid', how='left')
                a = np.nan_to_num(df_xdsegid['dist'], nan=1000)
                a[a<=graph_dist_threshold] = 1
                a[a>graph_dist_threshold] = 0
                distance_matrix[i] = a

                if i % 1000 == 0:
                    print('\t\tdone with {}/{}. Time elapsed: {}'.format(i, num_segs, time.time() - start_time))

            return distance_matrix
        else:
            raise ValueError('unrecognized graph distance type')

        #############
        ## Other methods that could be used in the future. But these need to be converted to the above style that
        ## outputs a numpy 2d array rather than a dataframe

        # elif method== 'dijkstra':
        #     for Seg in segment_list:
        #         graph_distance_dict[Seg] = dict(nx.single_source_dijkstra_path_length(graph, Seg, threshold))
        # elif method== 'dijkstra_weighted':
        #     for Seg in segment_list:
        #         graph_distance_dict[Seg] = dict(nx.single_source_dijkstra_path_length(graph,
        #                                                                               Seg,
        #                                                                               threshold,
        #                                                                               weight=weight_feature))
        # elif method== 'single_dijkstra_weighted':
        #     for Seg_i in segment_list:
        #         graph_distance_dict[Seg_i]=dict()
        #         for Seg_j in segment_list:
        #             print(Seg_i,Seg_j)
        #             graph_distance_dict[Seg_i][Seg_j] = nx.dijkstra_path_length(graph, Seg_i, Seg_j, weight_feature)
        # elif method== 'bellman_ford':
        #     for Seg in segment_list:
        #         graph_distance_dict[Seg] = dict(nx.single_source_bellman_ford_path_length(graph, Seg, threshold))
        # return  graph_distance_dict

    def create_rate_diff_matrix(self,
                                _df_data,
                                incident_rate_name):
        '''
        For every pair of segments, calcualte the difference in incident rates:
            |(Total_a - Total_b)| / Mean(Total_a, Total_b)
        adds this feature to a new dataframe
        '''
        _diff_rate_matrix = np.array([[np.nan] * len(_df_data)] * len(_df_data))
        for i,SegI in _df_data.iterrows():
            _diff_rate_matrix[:,i]=(2 * (_df_data[incident_rate_name] - SegI[incident_rate_name]) /
                                    (_df_data[incident_rate_name] + SegI[incident_rate_name])).abs().tolist()
        return _diff_rate_matrix


    def create_euclidean_connectivity_matrix(self,
                                             _df_data,
                                             threshold,
                                             unit_name):
        '''
        Returns a matrix of the euclidean connectivity of segments - segments are connected (1) if they are within
            <threshold> km of each other via euclidean distance, otherwise they are not connected (0)
        Note that segments are connected to themselves
        @param _df_data:
        @param threshold:
        @param unit_name:
        @return:
        '''

        start_time = time.time()
        print('\tstarting eucliean connectivity matrix creation')
        _euclidean_connectivity_matrix = _df_data[[unit_name,'center']].copy()
        _euclidean_connectivity_matrix = gpd.GeoDataFrame(_euclidean_connectivity_matrix,
                                                          geometry=_euclidean_connectivity_matrix['center'],
                                                          crs='epsg:4326')
        _euclidean_connectivity_matrix = _euclidean_connectivity_matrix.to_crs('EPSG:3310')
        _euclidean_connectivity_matrix['geometry'] = _euclidean_connectivity_matrix.buffer(threshold*1000/2) # TODO | why arithmetic?

        # perform a spatial join such that each entry in the resulting dataframe corresponds to segments that are
        #   within eachother's buffer zones - i.e. the distance b/w them is < threshold
        # doc: https://geopandas.org/docs/user_guide/mergingdata.html
        _joint_df = gpd.sjoin(_euclidean_connectivity_matrix[[unit_name,'geometry']],
                              _euclidean_connectivity_matrix[[unit_name,'geometry']],
                              how="inner",
                              op='intersects').reset_index()

        # create the connectivity matrix; 1 for each segment pair with distance < threshold
        _euclidean_connectivity_matrix = np.zeros((len(_df_data),len(_df_data)), dtype=int)
        for k in range(len(_joint_df)):
            i=_joint_df['index'].iloc[k]
            j=_joint_df['index_right'].iloc[k]
            _euclidean_connectivity_matrix[i,j]=1

        print('\teuclidean connectivity creatation time: {}'.format(time.time() - start_time))
        # pickle.dump(_euclidean_connectivity_matrix, open('_euclidean_connectivity_matrix.pkl', 'wb'))

        return _euclidean_connectivity_matrix


    def create_graph_connectivity_matrix(self,
                                         _df_data,
                                         _graph,
                                         threshold,
                                         unit_name,
                                         _save=False,
                                         _load=False,
                                         _save_loc='_graph_connectivity_matrix.pkl'):

        start_time = time.time()
        print('\tstarting graph connectivity matrix creation')

        graph_distance_matrix = None

        if _load:
            graph_distance_matrix = pickle.load(open(_save_loc, 'rb'))
        else:
            graph_distance_matrix = self.create_graph_connectivity_helper(_df_data[unit_name].tolist(),
                                                                          _graph,
                                                                          method='simple',
                                                                          path_length_threshold=100,
                                                                          graph_dist_threshold=threshold,
                                                                          data_df=_df_data)

            if _save:
                pickle.dump(graph_distance_matrix, open(_save_loc, 'wb'))

        print('\tgraph connectivity creatation time: {}'.format(time.time() - start_time))
        return graph_distance_matrix

    def create_rate_diff_connectivity_matrix(self,
                                             _df_data,
                                             threshold,
                                             incident_rate_name):

        _rate_diff_matrix = self.create_rate_diff_matrix(_df_data, incident_rate_name=incident_rate_name)

        # convert to connectivity matrix
        _rate_diff_matrix[_rate_diff_matrix<=threshold]=1
        _rate_diff_matrix[_rate_diff_matrix>threshold]=0

        return _rate_diff_matrix

    def get_connected_components(self,
                                 _graph_connectivity_matrix):
        '''
        Find the connected components of the provided distance matrix.
        @param _graph_connectivity_matrix:
        @return:
        '''

        connected_components=scipy.sparse.csgraph.connected_components(_graph_connectivity_matrix)
        connected_components_df=pd.DataFrame({'Component_Index':connected_components[1]})
        return connected_components_df

    ##########################################
    ## Plotting code

    def Plot(self,
             DF_inrix_accident,
             unit_name,
             df_edges,
             incident_rate_name,
             Show_Links_Tag=False):
        '''
        This function plots
        1) the center of the segments
        2) color code based on the cluster the belong to
        3) optionally you can show the connection between the segments
        '''

        #######################
        ## Plotting the center of the Segments using points

        #adding the Cluster name to the DF_inrix
        DF=DF_inrix_accident.copy()
        full_index = []
        counter = 0
        for i, row in DF.iterrows():
            if isinstance(row.geometry, shapely.geometry.collection.GeometryCollection):
                print(i, row.geometry)
                counter += 1
            else:
                full_index.append(i)

        DF = DF.loc[full_index]
        DF['center_lat'],DF['center_lon']=zip(*DF.apply(lambda row: (row.center.coords[0][1],row.center.coords[0][0]),axis=1))
        DF=DF[DF[incident_rate_name]>=1]
        color_discrete_map =dict(zip([str(i) for i in np.sort(DF['cluster_label'].unique())],list(px.colors.qualitative.Light24)))
        DF['cluster_label']=DF['cluster_label'].astype('str')
        fig = px.scatter_mapbox(DF, lat="center_lat", lon="center_lon",
                                opacity=.5, color='cluster_label',
                                size=[1.5]*len(DF),hover_name=unit_name,
                                hover_data =['Degree',incident_rate_name,'index'],
                                color_discrete_map=color_discrete_map)
        lats = [];lons = [];colors= [];texts=[]

        if Show_Links_Tag:
            for i,row in df_edges.iterrows():
                lats = np.append(lats, row.source_center.coords[0][1])
                lats = np.append(lats, row.targe_center.coords[0][1])
                lons = np.append(lons, row.source_center.coords[0][0])
                lons = np.append(lons, row.targe_center.coords[0][0])
                texts= np.append(texts, row.source)
                texts= np.append(texts, row.target)
                colors = np.append(colors, ['gray']*3)
                lats = np.append(lats, None);    lons = np.append(lons, None); texts = np.append(texts, None)
            fig.add_trace(go.Scattermapbox(lon=lons,
                                           lat=lats ,
                                           mode = 'lines',
                                           line = dict(width = 1,color = 'gray'),
                                           hovertext     =texts))
            fig.update_layout(showlegend=False)
        fig.update_layout(mapbox_style="carto-darkmatter",mapbox_zoom=7.5, mapbox_center_lat = 36, mapbox_center_lon=-86)
        fig.show()
        # fig.write_image('test_map.html')

        '''
        #%%Plotting Using Geopandas
        #1) Plotting the center of the Segments using points
        fig, ax = plt.subplots(figsize = (15,15)) 
        GDF_inrix=DF_inrix.copy()
        GDF_inrix=gpd.GeoDataFrame(DF_inrix, geometry=DF_inrix.center)
        GDF_inrix.plot(column='cluster_label',ax=ax)
        ax.set_title('center of the Segments color coded by cluster number')
        
        #2)Plotting the links (the connection between the center of the segments) using Geopandas using lines
        fig, ax = plt.subplots(figsize = (15,15)) 
        GDF_adj=GDF_adj.copy()
        GDF_adj = gpd.GeoDataFrame(DF_adj, geometry=DF_adj.Line)
        GDF_adj.plot(column='Percent_diff_Total_Number_Incidents',ax=ax)
        ax.set_title('Link between center of the segments color coded based on the gradiant of the rates')
        '''


    def plot_dendrogram(self,
                        model,
                        linkage,
                        **kwargs):

        hierarchy.set_link_color_palette(list(px.colors.qualitative.Light24))
        fig, axes = plt.subplots(figsize=(10, 5))

        # plot the top three levels of the dendrogram
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, truncate_mode='level', p=3,ax=axes)

        axes.set_xlabel("Number of points in node (or index of point if no parenthesis).")
        #plt.axhline(y=distance_threshold, color='r', linestyle='--')
        plt.suptitle(linkage)
        plt.show()
        # plt.savefig('new_dendogram.png')
