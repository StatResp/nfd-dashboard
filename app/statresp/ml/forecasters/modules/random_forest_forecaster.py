from statresp.ml.forecasters.modules.base import Forecaster
import json
import pickle
import time
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from patsy import dmatrices
import numpy as np
from sklearn.model_selection import PredefinedSplit

###########################
## hard coded hyperparamter choices. Ensure that you have one entry in the array for each cluster
CV_Number=5
class RF(Forecaster):
    

    def __init__(self, model_type,metadata_current):
        
        self.model_type = model_type
        self.model_params = {}
        self.model_stats = {}
        self.model_threshold= {}
        self.metadata=metadata_current   
        self.features=metadata_current['features_ALL']
        self.name=metadata_current['Model_Full_Name']#  'RF'+'+'+self.metadata['Cluster_Name']

        
        self.best_params = {}

    def hyper_param_name(self, metadata, model_hyperparams_dict):
        
        '''
        name = 'hyper_' + metadata['current_model']['Name']
        name = name + '_' + model_hyperparams_dict['class_weights']
        name = name + '_' + str(metadata['start_time_train'])+'to'+str(metadata['end_time_test'])
        name = name + '.pkl'
        name = name.replace(':', '-')
        name = name.replace(' ', '')
        '''
        metadata['current_model']['hypername'] = 'RF'+model_hyperparams_dict['class_weights']+self.metadata['Data_Name']




    def fit(self, df_train, df_verif, metadata, sampling_type, model_hyperparams_address):
        print('\nRF has started...')
        model_hyperparams_dict = json.load(open(model_hyperparams_address, 'rb'))

        # fit method used depends on the hyperparameter fit type from metadata
        if model_hyperparams_dict['search_type'] == 'None':
            self.fit_no_search(df_train, df_verif, metadata, sampling_type, model_hyperparams_dict)
        elif model_hyperparams_dict['search_type'] == 'random_grid':
            self.fit_random_grid_search(df_train, df_verif, metadata, sampling_type, model_hyperparams_dict)
        elif model_hyperparams_dict['search_type'] == 'full_grid':
            self.fit_full_grid_search(df_train, df_verif, metadata, sampling_type, model_hyperparams_dict)


    def fit_no_search(self, df_train, df_verif, metadata, sampling_type, model_hyperparams_dict):

        print('no hyperparam search')
        features=self.features
        expr = self.get_regression_expr(metadata,features)

        ##BalanceFactor=Balance_Adjuster(df_train,metadata)
        clusters = sorted(df_train.cluster_label.unique())

        _class_weight_type = model_hyperparams_dict['class_weights']
        if _class_weight_type == 'None':
            _class_weight_type = None

        cluster_classifiers = dict()
        for cluster in clusters:
            cluster_classifiers[cluster] = RandomForestClassifier(
                        n_estimators=model_hyperparams_dict['hyperparams'][str(cluster)]['n_estimators'],
                        max_depth=model_hyperparams_dict['hyperparams'][str(cluster)]['max_depth'],
                        min_samples_split=model_hyperparams_dict['hyperparams'][str(cluster)]['min_samples_split'],
                        min_samples_leaf=model_hyperparams_dict['hyperparams'][str(cluster)]['min_samples_leaf'],
                        class_weight=_class_weight_type,
                        n_jobs=model_hyperparams_dict['number_of_threads'],
                        random_state=metadata['seed_number'],
                    )

        for temp_cluster in clusters:
            #print('cluster number: ',temp_cluster)
            self.model_params[temp_cluster] = cluster_classifiers[temp_cluster]

            df_cluster_train = df_train.loc[df_train.cluster_label == temp_cluster]
            df_cluster_verif = df_verif.loc[df_verif.cluster_label == temp_cluster]
            y_train, x_train = dmatrices(expr, df_cluster_train, return_type='dataframe')
            y_verif, x_verif = dmatrices(expr, df_cluster_verif, return_type='dataframe')



            #_my_y_train = np.array(_my_y_train)
            #_my_y_train = np.ravel(_my_y_train)
  
            y_train = np.array(y_train)
            y_train = np.ravel(y_train)
            

        

            #print('\ttraining the model')
            start_time = time.time()
            self.model_params[temp_cluster].fit(x_train, y_train)
            #print('\t\tfit time: {}'.format(time.time() - start_time))

            #print('\ttuning threshold')
            y_verif = np.array(y_verif)
            y_verif = np.ravel(y_verif)

            y_hat_verif_probs = self.model_params[temp_cluster].predict_proba(x_verif)
            precision, recall, thresholds = precision_recall_curve(y_true=y_verif,
                                                                   probas_pred=y_hat_verif_probs[:,1])
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.nanargmax(fscore)
            print('For Cluster=%.0f, Best Threshold=%f, F-Score=%.5f' % (temp_cluster, thresholds[ix], fscore[ix]))
            #print('\tBest Threshold=%f, F-Score=%.5f' % (thresholds[ix], fscore[ix]))
            if np.isnan(fscore[ix])==True:
                self.model_threshold[temp_cluster] = 0.5
            else:
                self.model_threshold[temp_cluster] = thresholds[ix]

        self.update_model_stats(clusters)


    def fit_random_grid_search(self, df_train, df_verif, metadata, sampling_type, model_hyperparams_dict):
        print('randomized grid search')
        features=self.features
        expr = self.get_regression_expr(metadata,features)
        clusters = sorted(df_train.cluster_label.unique())

        _class_weight_type = model_hyperparams_dict['class_weights']
        if _class_weight_type == 'None':
            _class_weight_type = None

        for curr_cluster in clusters:
            #print('cluster number: ',curr_cluster)
            estimator = RandomForestClassifier()

            df_cluster_train = df_train.loc[df_train.cluster_label == curr_cluster]
            df_cluster_verif = df_verif.loc[df_verif.cluster_label == curr_cluster]
            y_train, x_train = dmatrices(expr, df_cluster_train, return_type='dataframe')
            y_verif, x_verif = dmatrices(expr, df_cluster_verif, return_type='dataframe')


            y_train = np.array(y_train)
            y_train = np.ravel(y_train)

            hyperparam_search_grid = {
                'n_estimators': model_hyperparams_dict['hyperparam_grids'][str(curr_cluster)]['n_estimators'],
                'max_depth': model_hyperparams_dict['hyperparam_grids'][str(curr_cluster)]['max_depth'],
                'min_samples_split': model_hyperparams_dict['hyperparam_grids'][str(curr_cluster)]['min_samples_split'],
                'min_samples_leaf': model_hyperparams_dict['hyperparam_grids'][str(curr_cluster)]['min_samples_leaf'],
                'bootstrap': [True],
                'max_samples': [0.8],
                'max_features': ['auto'],
                'oob_score': [False],
                'class_weight': [_class_weight_type]
            }

            print('\tBegin Grid Search')
            start_time = time.time()
            rf_random_search = RandomizedSearchCV(estimator = estimator,
                                                  param_distributions = hyperparam_search_grid,
                                                  n_iter = model_hyperparams_dict['grid_n_iterations'],
                                                  cv=CV_Number,
                                                  verbose=2,
                                                  random_state=metadata['seed_number'],
                                                  n_jobs = model_hyperparams_dict['number_of_threads'],
                                                  scoring=model_hyperparams_dict['grid_scoring'])

            rf_random_search.fit(x_train, y_train)
            print('\t\tsearch time: {}'.format(time.time() - start_time))

            print('\t\tbest params for cluster {}: {}'.format(curr_cluster, rf_random_search.best_params_))
            self.best_params[curr_cluster] = rf_random_search.best_params_
            self.model_params[curr_cluster] = rf_random_search.best_estimator_

            # self.feature_importances[temp_cluster] = dict(zip(x_train.columns, self.model_params[temp_cluster].feature_importances_))
            # print(self.feature_importances[temp_cluster])

            #print('\ttuning threshold')
            y_verif = np.array(y_verif)
            y_verif = np.ravel(y_verif)

            y_hat_verif_probs = self.model_params[curr_cluster].predict_proba(x_verif)
            precision, recall, thresholds = precision_recall_curve(y_true=y_verif,
                                                                   probas_pred=y_hat_verif_probs[:,1])
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.nanargmax(fscore)
            print('For Cluster=%.0f, Best Threshold=%f, F-Score=%.5f' % (curr_cluster, thresholds[ix], fscore[ix]))
            #print('\tBest Threshold=%f, F-Score=%.5f' % (thresholds[ix], fscore[ix]))
            if np.isnan(fscore[ix])==True:
                self.model_threshold[curr_cluster] = 0.5
            else:
                self.model_threshold[curr_cluster] = thresholds[ix]

        self.update_model_stats(clusters)
        self.hyper_param_name(metadata, model_hyperparams_dict)
        #pickle.dump(self.best_params, open(metadata['current_model']['hypername'], 'wb'))


    def fit_full_grid_search(self, df_train, df_verif, metadata, sampling_type, model_hyperparams_dict):
        print('full grid search')
        features=self.features
        expr = self.get_regression_expr(metadata,features)

        clusters = sorted(df_train.cluster_label.unique())

        _class_weight_type = model_hyperparams_dict['class_weights']
        if _class_weight_type == 'None':
            _class_weight_type = None

        for curr_cluster in clusters:
            #print('cluster number: ',curr_cluster)
            estimator = RandomForestClassifier()

            df_cluster_train = df_train.loc[df_train.cluster_label == curr_cluster]
            df_cluster_verif = df_verif.loc[df_verif.cluster_label == curr_cluster]
            y_train, x_train = dmatrices(expr, df_cluster_train, return_type='dataframe')
            y_verif, x_verif = dmatrices(expr, df_cluster_verif, return_type='dataframe')


            y_train = np.array(y_train)
            y_train = np.ravel(y_train)


            hyperparam_search_grid = {
                'n_estimators': model_hyperparams_dict['hyperparam_grids'][str(curr_cluster)]['n_estimators'],
                'max_depth': model_hyperparams_dict['hyperparam_grids'][str(curr_cluster)]['max_depth'],
                'min_samples_split': model_hyperparams_dict['hyperparam_grids'][str(curr_cluster)]['min_samples_split'],
                'min_samples_leaf': model_hyperparams_dict['hyperparam_grids'][str(curr_cluster)]['min_samples_leaf'],
                'bootstrap': [True],
                'max_samples': [0.8],
                'max_features': ['auto'],
                'oob_score': [False],
                'class_weight': [_class_weight_type]
            }

            print('\tBegin Grid Search')
            start_time = time.time()
            rf_grid_search = GridSearchCV(estimator = estimator,
                                          param_grid = hyperparam_search_grid,
                                          cv=CV_Number,
                                          verbose=2,
                                          random_state=metadata['seed_number'],
                                          n_jobs = model_hyperparams_dict['number_of_threads'],
                                          scoring=model_hyperparams_dict['grid_scoring'],
                                          )

            rf_grid_search.fit(x_train, y_train)
            print('\t\tsearch time: {}'.format(time.time() - start_time))

            print('\t\tbest params for cluster {}: {}'.format(curr_cluster, rf_grid_search.best_params_))
            self.best_params[curr_cluster] = rf_grid_search.best_params_
            self.model_params[curr_cluster] = rf_grid_search.best_estimator_



            # self.feature_importances[temp_cluster] = dict(zip(x_train.columns, self.model_params[temp_cluster].feature_importances_))
            # print(self.feature_importances[temp_cluster])

            #print('\ttuning threshold')
            y_verif = np.array(y_verif)
            y_verif = np.ravel(y_verif)

            y_hat_verif_probs = self.model_params[curr_cluster].predict_proba(x_verif)
            precision, recall, thresholds = precision_recall_curve(y_true=y_verif,
                                                                   probas_pred=y_hat_verif_probs[:,1])
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.nanargmax(fscore)
            print('For Cluster=%.0f, Best Threshold=%f, F-Score=%.5f' % (curr_cluster, thresholds[ix], fscore[ix]))
            #print('\tBest Threshold=%f, F-Score=%.5f' % (thresholds[ix], fscore[ix]))
            if np.isnan(fscore[ix])==True:
                self.model_threshold[curr_cluster] = 0.5
            else:
                self.model_threshold[curr_cluster] = thresholds[ix]

        self.update_model_stats(clusters)
        self.hyper_param_name(metadata, model_hyperparams_dict)
        #pickle.dump(self.best_params, open(metadata['current_model']['hypername'], 'wb'))



    def prediction(self, df, metadata):

        df_complete_predicted = df.copy()
        df_complete_predicted['predicted'] = 0
        df_complete_predicted['Sigma2'] = 0
        df_complete_predicted['ancillary'] = 0
        features = self.features

        clusters = sorted(df.cluster_label.unique())
        for temp_cluster in clusters:
            #print('cluster number: ',temp_cluster)
            prob_preds = self.model_params[temp_cluster].predict_proba(df[df.cluster_label == temp_cluster][features])
            df_complete_predicted.loc[df.cluster_label == temp_cluster,'predicted'] = [_[1] for _ in prob_preds]
            df_complete_predicted.loc[df.cluster_label==temp_cluster,'threshold']=self.model_threshold[temp_cluster]
            df_complete_predicted['predicted_TF']=df_complete_predicted['predicted']>df_complete_predicted['threshold']


        if metadata['pred_name_TF'] in df.columns:
            df_complete_predicted['error'] = df_complete_predicted[metadata['pred_name_TF']] - df_complete_predicted['predicted']
            MSE_all,  MSE = self.MSE(df_complete_predicted, metadata)

            return [df_complete_predicted, [np.nan for _ in clusters], np.nan,MSE_all, MSE]
        else:
            return [df_complete_predicted, None, None, None, None]


    def get_regression_expr(self, metadata,features):
        """
        Creates regression expression in the form of a patsy expression
        @param features: x features to regress on
        @return: string expression
        """
        expr = metadata['pred_name_TF']+'~'
        for i in range(len(features)):
            # patsy expects 'weird columns' to be inside Q
            if ' ' in features[i]:
                expr += "Q('" + features[i] + "')"
            else:
                expr += features[i]
            if i != len(features) - 1:
                expr += '+'
        expr  += '-1'
        return expr

    def update_model_stats(self, clusters):
        # smv:checked
        """
        Store the the summation of log likelihood of the training set, AIC value.
        @return: _
        """
        self.model_stats['train_likelihood'] = np.nan
        self.model_stats['aic'] = np.nan
        self.model_stats['train_likelihood_all']=[np.nan for _ in clusters]
        self.model_stats['aic_all']=[np.nan for _ in clusters]


    def MSE(self, df, metadata):
        #smv:checked
        """
        Return the Mean Square Error (MSE) of model for the incidents in df for prediction
        @df: dataframe of incidents to calculate likelihood on
        @param metadata: dictionary with feature names, cluster labels.
        @return: likelihood value for each sample and the total summation as well the updated df which includes llf
        """
        df['error2']=df['error']**2
        MSE=np.mean(df['error2'])
        MSE_all=df[['error2','cluster_label']].groupby(['cluster_label'], as_index=False).mean().sort_values(by='cluster_label', ascending=True)
        return [(MSE_all['error2'].values).tolist(),MSE]

    def Likelihood(self):
        pass


