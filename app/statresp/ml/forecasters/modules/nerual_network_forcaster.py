"""
@Author - Ayan Mukhopadhyay
Classification Forecaster -- Inherits from Forecaster class
"""
from statresp.ml.forecasters.modules.base import Forecaster
import numpy as np
import pandas as pd
from copy import deepcopy
from patsy import dmatrices
from sklearn import svm
from sklearn import metrics
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
import os

def create_default_meta(df, static_features=None):
    """
    Creates default set of metadata if user supplied data is missing
    @param df: dataframe of incidents
    @param static_features: set of static features used in clustering
    @return: metadata dictionary
    """
    metadata = {'start_time_train': df['time'].min(), 'end_time_train': df['time'].max()}
    if static_features is None:
        static_features = list(df.columns)
        if 'cluster_label' in static_features:
            static_features.remove('cluster_label')
    metadata['features_ALL'] = static_features
    return metadata



#%%

class neuralNet(Forecaster):
    
    
    # This is the main class for neural network model
    def __init__(self, model_type,metadata_current):
        
        self.model_type = model_type
        self.model_params = {}
        self.model_stats = {}
        self.model_threshold= {}
        self.metadata=metadata_current
        self.features=metadata_current['features_ALL']
        try:
            self.name=metadata_current['Model_Full_Name']#'NN'+'+'+self.metadata['Cluster_Name']
        except:
            self.name==None
        self.submodels=dict()
        

        

    def fit(self, df_train,df_verif, metadata=None, sampling_name='No_Resample'):
        print('\nNN has started...')
        if metadata is None:
            metadata = create_default_meta(df_train)

        # get regression expression
        features=self.features
        expr = self.get_regression_expr(metadata, features)        

        clusters = sorted(df_train.cluster_label.unique())
        for temp_cluster in clusters:
            #print('cluster number: ', temp_cluster)
            df_cluster_train = df_train.loc[df_train.cluster_label == temp_cluster]
            df_cluster_verif = df_verif.loc[df_verif.cluster_label == temp_cluster]
            y_train, x_train = dmatrices(expr, df_cluster_train, return_type='dataframe')
            y_verif, x_verif = dmatrices(expr, df_cluster_verif, return_type='dataframe')




            # create simple neural net model
            nn_model= self.create_nn_architecture(x_train.shape[1],metadata['seed_number'])
            # compile the model
            nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # fit model
            nn_model.fit(x_train, y_train, epochs=2, batch_size=100)

            # adjust threhold
            y_train_hat = nn_model.predict(x_verif)
            precision, recall, thresholds = precision_recall_curve(y_verif, y_train_hat)
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.nanargmax(fscore)
            print('For Cluster=%.0f, Best Threshold=%f, F-Score=%.5f' % (temp_cluster, thresholds[ix], fscore[ix]))
            if np.isnan(fscore[ix]) == True:
                self.model_threshold[temp_cluster] = 0.5
            else:
                self.model_threshold[temp_cluster] = thresholds[ix]

            self.model_params[temp_cluster] = nn_model
            ##TODO: CHANGE PATH
            self.submodels[temp_cluster]='NN'+str(temp_cluster)+self.metadata['Data_Name']
            #Dir='output/train/models/SubNNs/'+'+'+self.submodels[temp_cluster]+'.h5'
            #nn_model.save(Dir)            

            #pickle.dump(self.model_params[temp_cluster], open('Address.sav', 'wb'))

        self.update_model_stats()
        print('Finished Learning {} model.'.format(self.model_type))


    def create_nn_architecture(self, input_shape,seed_number=0):
        np.random.seed(seed_number)
        tf.random.set_seed(seed_number)
        
        nn_model = Sequential()
        nn_model.add(Dense(input_shape * 2, activation='relu'))
        nn_model.add(Dense(input_shape, activation='relu'))
        nn_model.add(Dense(input_shape, activation='relu'))
        nn_model.add(Dense(1, activation='sigmoid'))
        return nn_model

    def prediction(self, df, metadata):
        df_complete_predicted = df.copy()
        df_complete_predicted['predicted'] = 0
        features = self.features
        clusters = sorted(df.cluster_label.unique())
        for temp_cluster in clusters:
            #print('cluster number: ', temp_cluster)
            # add the threshold as a column : not sure why this is being done. At the moment, doing what GLM
            # forecaster does
            df_complete_predicted.loc[df.cluster_label == temp_cluster, 'threshold'] = self.model_threshold[
                temp_cluster]

            df_complete_predicted.loc[df_complete_predicted.cluster_label == temp_cluster, 'predicted'] = \
            self.model_params[temp_cluster].predict(
                df_complete_predicted[df_complete_predicted.cluster_label == temp_cluster][features])
            df_complete_predicted['predicted_TF'] = (df_complete_predicted['predicted'] > df_complete_predicted['threshold']).astype(int)
            # df_cluster['predicted'] = (df_complete_predicted['predicted']>0.5).astype(int)
            # df_complete_predicted.loc[df.cluster_label==temp_cluster,'predicted']=deepcopy(df_cluster['predicted'])
            
            '''
            dump_name = 'df_predicted_' + str(temp_cluster) + '.pickle'
            with open(dump_name, 'wb') as f:
                pickle.dump(df_complete_predicted, f)
            print('Dumped pickle file for cluster {}'.format(temp_cluster))
            '''
        if metadata['pred_name_TF'] in df.columns:
            df_complete_predicted['error'] = df_complete_predicted[metadata['pred_name_TF']] - df_complete_predicted[
                'predicted']

            MSE_all, MSE = self.MSE(df_complete_predicted, metadata)
            return [df_complete_predicted, None, None, MSE_all, MSE]
        else:
            return [df_complete_predicted, None, None, None, None]

    def get_regression_expr(self, metadata, features):
        """Creates regression expression in the form of a patsy expression
        @param features: x features to regress on
        @return: string expression
        """
        expr = metadata['pred_name_TF'] + '~'
        for i in range(len(features)):
            # patsy expects 'weird columns' to be inside Q
            if ' ' in features[i]:
                expr += "Q('" + features[i] + "')"
            else:
                expr += features[i]
            if i != len(features) - 1:
                expr += '+'
        expr += '-1'
        return expr

    def MSE(self, df, metadata):
        # smv:checked
        """
        Return the Mean Square Error (MSE) of model for the incidents in df for prediction
        @df: dataframe of incidents to calculate likelihood on
        @param metadata: dictionary with feature names, cluster labels.
        @return: likelihood value for each sample and the total summation as well the updated df which includes llf
        """
        df['error2'] = df['error'] ** 2
        MSE = np.mean(df['error2'])
        MSE_all = df[['error2', 'cluster_label']].groupby(['cluster_label'], as_index=False).mean().sort_values(
            by='cluster_label', ascending=True)
        return [(MSE_all['error2'].values).tolist(), MSE]

    def update_model_stats(self):
        # smv:checked
        """
        Store the the summation of log likelihood of the training set, AIC value.
        @return: _
        """
        self.model_stats['train_likelihood'] = None
        self.model_stats['aic'] = None
        self.model_stats['train_likelihood_all'] = None
        self.model_stats['aic_all'] = None
        
        
    def saveNN(self,Address):
        model=self
        model_params=dict()
        #model_params=deepcopy(self.model_params)
 
        #directory='output/train/models/'+'SubNNs/'
        directory=Address+'SubNNs/'
        if not os.path.exists(directory):
             os.makedirs(directory)         
 
    
        for i in model.model_params:
             Dir=directory+str(i)+model.name+'.h5'
             #print(Dir)
             #model.model_params[i]=deepcopy(self.model_params[i])
             model_params[i]=model.model_params[i]
             model.model_params[i].save(Dir) 
             model.model_params[i]=None
        #del model.model_params
        #delattr(NN, 'model_params')
        pickle.dump(model, open(Address+model.name+'.sav', 'wb'))
        for i in model.model_params:
            model.model_params[i]=model_params[i]
        
    def loadNN( self, Address):
        # model = pickle.load(open('output/train/models/'+metadata['model_to_predict'], 'rb'))
        model= pickle.load(open(Address, 'rb'))
        Address=('/').join(Address.split('/')[0:-1])
        #print(Address)
        for i in model.model_params:
             Dir=Address+'/SubNNs/'+str(i)+model.name+'.h5'
             #print(Dir)
             model.model_params[i] = tf.keras.models.load_model(Dir)
        return model       
        
        '''
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")        
        '''
        