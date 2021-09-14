"""
Base Class for all forecasters. All classes must implement the following methods:

1. fit -- fit a regression model to data
2. predict -- sample instances based on the forecasting model
3. get_regression_expr -- get a patsy expression for the regression.
4. update_model_stats -- store the model likelihood, AIC score for easy access
"""
from statresp.ml.forecasters.modules.base import Forecaster
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd

class Naive_Model(Forecaster):
#This is the main GLM_Model calss, and all GLM fall into this category 




    def __init__(self,model_type,metadata_current):
        self.model_type = model_type
        self.model_params = {}
        self.model_stats = {}
        self.model_threshold= {}
        self.metadata=metadata_current
        self.name=metadata_current['Model_Full_Name']#'Naive'+'+'+self.metadata['Data_Name']
        
        
    def __Threshold_Adjuster(self,model, df_verif,metadata):
            y_verif_hat=pd.merge(df_verif[metadata['unit_name']],pd.DataFrame.from_dict(model, orient='index').reset_index().rename(columns={'index':metadata['unit_name'] }), left_on= metadata['unit_name'] ,right_on=  metadata['unit_name'], how='left'  )
            y_verif_hat=y_verif_hat['predicted']
            #y_verif_hat= model.predict(df_verif)
            precision, recall, thresholds = precision_recall_curve(df_verif[metadata['pred_name_TF']], y_verif_hat)
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.nanargmax(fscore)
            #ix = np.argmax(fscore)

            print('Best Threshold=%f, F-Score=%.5f' % (thresholds[ix], fscore[ix]))
            #print('     Threshold=%f, F-Score=%.5f' % (thresholds[ix-1], fscore[ix-1]))
            #print('     Threshold=%f, F-Score=%.5f' % (thresholds[ix+1], fscore[ix+1]))
            if np.isnan(fscore[ix])==True:
                self.model_threshold = 0.5
            else:
                self.model_threshold = thresholds[ix]

    def fit(self, df_train, df_verif, metadata=None,resampling_type='No_Resample'):
            """
            Fits regression model to data
            @param df: dataframe of incidents in regression format
            @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                            see github documentation for details
            @return: _
            """
            DF=df_train[[metadata['unit_name'],metadata['pred_name_TF']]].groupby(metadata['unit_name']).agg({metadata['pred_name_TF']: ['count','mean']})
            DF.columns=['count', 'predicted']
            #DF=DF.reset_index()
            model=DF.to_dict(orient='index')
            self.__Threshold_Adjuster(model, df_verif,metadata )
            self.model_params= model
               
            self.update_model_stats()
            print('Finished Learning {} model.'.format(self.model_type))
        
    def prediction(self, df, metadata):
            """
            Predicts E(y|x) for a set of x, where y is the concerned dependent variable.
            @param df: dataframe of incidents in regression format
            @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                            see github documentation for details
            @return: updated dataframe with predicted values, Sigma2, ancillary and information regarding llf and MSE
            """
            DF=pd.DataFrame.from_dict(self.model_params, orient='index').reset_index().rename(columns={'index':metadata['unit_name'] })
            df_complete_predicted=df.copy()
            df_complete_predicted['predicted_TF']=None
            df_complete_predicted=pd.merge(df_complete_predicted,DF, left_on= metadata['unit_name'] ,right_on=  metadata['unit_name'], how='left'  )
        
            df_complete_predicted['threshold']=self.model_threshold

                
            df_complete_predicted['predicted_TF']=df_complete_predicted['predicted']>df_complete_predicted['threshold']        

            if (metadata['pred_name_TF'] in  df.columns): #| (metadata['pred_name_Count'] in  df.columns):
                df_complete_predicted['error']=df_complete_predicted[metadata['pred_name_TF']]-df_complete_predicted['predicted']
                test_likelihood_all,  test_likelihood, df = None, None, None
                MSE_all,  MSE = self.MSE(df_complete_predicted, metadata)     
                return [df_complete_predicted,test_likelihood_all,test_likelihood,MSE_all, MSE]
            else:
                return [df_complete_predicted,None,None,None,None]




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
        return [[MSE],MSE]