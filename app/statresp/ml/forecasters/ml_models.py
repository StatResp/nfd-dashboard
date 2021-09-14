"""
@Author - Sayyed Mohsen Vazirizade and Ayan Mukhopadhyay
Parent File for all regression models for incident prediction.
Currently supports -
1. Poisson Regression (Count Based)
2. Negative Binomial Regression (Count Based)
3. Parametric Survival Regression (Time Based)
"""
from statresp.ml.forecasters.modules.reg_forecaster import GLM_Model
from statresp.ml.forecasters.modules.naive import Naive_Model
from statresp.ml.forecasters.modules.nerual_network_forcaster import neuralNet
from statresp.ml.forecasters.modules.svm_forcaster import SVM
from statresp.ml.forecasters.modules.random_forest_forecaster import RF
from statresp.ml.forecasters.utils import  update_meta_features
from statresp.utils.read_data import Read_data
from statresp.utils.main_utils import add_to_current_metadata, add_to_metadata_prediction
from statresp.ml.forecasters.utils import  Conf_Mat, Results_Generator
from statresp.dataprep.traintest.traintest import create_predict_dataset
from statresp.utils.read_data import prepare_merged_dataset, check_trainmodel_exist
from statresp.ml.forecasters.modules.base import Forecaster
import numpy as np


def ml_train_test(metadata):
    '''
    This function imports df_train, df_verif, df_test
    The uses uses df_train, df_verif, the learn function to train the intented model.
    It also uses df_test to provide some preduction results.  

    Parameters
    ----------
    metadata : dictionary
        DESCRIPTION.

    Returns
    -------
    model : class model
        This is the model trained and can be used.
    df_test : dataframe
        The test data with the predicted values by the model are added to it..
    performance : dictionary
        This includes vairous types of performance metrics.
    metadata : dictionary
        The updated metadata.
    '''
    np.random.seed(metadata['seed_number']) 
    #%%reading the data
    df_train= Read_data(metadata['current_model']['Data_Name'], directory=metadata['Input_Address']+"/resampled_clustered_traintestsplit/", Type='train')
    df_verif= Read_data(metadata['current_model']['Data_Name'], directory=metadata['Input_Address']+"/resampled_clustered_traintestsplit/", Type='validation')
    df_test= Read_data(metadata['current_model']['Data_Name'], directory=metadata['Input_Address']+"/resampled_clustered_traintestsplit/", Type='test')
    metadata['current_model']['features_ALL'] = update_meta_features(metadata['current_model']['features'], df_features=df_train.columns.tolist(), cat_col=metadata['cat_features'])
    #%%train
    if metadata['rewirte_Tag'] == False and check_trainmodel_exist(metadata['current_model']['Model_Full_Name'],
                                                                       metadata['Output_Address']):
        print('{} exists and no need to retrain.'.format(metadata['current_model']['Model_Full_Name']))
        model=Forecaster()
        model=model.load(metadata['Output_Address']+'/trained_models/models/'+metadata['current_model']['Model_Full_Name']+'.sav')
    else:
        print('{} does not exist. Training process has started: '.format(metadata['current_model']['Model_Full_Name']))
        model= learn(df_train, df_verif, metadata)
        model.save(metadata['Output_Address']+'/trained_models/models/')
    #%%test
    df_test, test_likelihood_all,test_likelihood,test_MSE_all,test_MSE = model.prediction(df=df_test, metadata=metadata)
    #%%performance
    Conf_Matrix = Conf_Mat(df_test, metadata, current_model=metadata['current_model'])
    #Conf_Matrix = Conf_Mat(df_pred, meta, model_name=metadata['models'][m])
    performance=Results_Generator(metadata['current_model']['model_type'], model, Conf_Matrix, test_likelihood, test_likelihood_all, None, None,test_MSE, test_MSE_all, None,None)
    return model,df_test,performance,metadata        
    




def ml_predict(metadata,model,df_merged=None):
    '''
    This function calls df_predict dataframe and use the provided model to predict the likelihood of incidents. 

    Parameters
    ----------
    metadata : Dictionary
        metadata.
    model : class model
        This is the model trained and can be used.

    Returns
    -------
    model : class model
        This is the model trained and can be used. It is not neccessary to return this but for sake of compatibility with ml_train_test it is returned
    df_test : dataframe
        The test data with the prediction added to it.
    performance : dictionary
        This includes vairous types of performance metrics.
    metadata : dictionary
        metadata. It is not neccessary to return this but for sake of compatibility with ml_train_test it is returned


    '''
    
    #df_predict= Read_data(model.metadata['Data_Name'],metadata['Output_Address']+"/preparation/",'predict')
    if metadata['predict_data_from_csv_tag']==True:
        create_predict_dataset(metadata=metadata)
    else:
        if df_merged is None:
            df_merged = prepare_merged_dataset(metadata)
        df_predict=create_predict_dataset(df=df_merged, metadata=metadata,model=model)
        df_predict, pred_likelihood_all,pred_likelihood,pred_MSE_all,pred_MSE = model.prediction(df=df_predict, metadata=metadata)
    #%%performance
    Conf_Matrix = Conf_Mat(df_predict, metadata, current_model=model.metadata)
    performance=Results_Generator(model.metadata['model_type'], model, Conf_Matrix, None, None, pred_likelihood, pred_likelihood_all,None, None, pred_MSE,pred_MSE_all)

    return model,df_predict,performance
    
    
    
def learn(df_train, df_verif, metadata):
    '''
    This function chooses the propel model to train. The available models are RF, NN, LR, and ZIP

    Parameters
    ----------
    df_train : TYPE
        DESCRIPTION.
    df_verif : TYPE
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    model : class model
        This is the model trained and can be used.

    '''
    

    if metadata['current_model']['model_type'] in metadata['GLM_Models']:
        model = GLM_Model(metadata['current_model']['model_type'],metadata['current_model'])
        model.fit(df_train,df_verif, metadata)
    elif metadata['current_model']['model_type']== 'Survival_Regression':
        model = Survival_Model()
        model.fit(df_train, metadata)
    elif metadata['current_model']['model_type']== 'SVM':
        model = SVM(metadata['current_model']['model_type'])
        model.fit(df_train, metadata)

    elif metadata['current_model']['model_type'] == 'NN':
        model = neuralNet(metadata['current_model']['model_type'],metadata['current_model'])
        model.fit(df_train,df_verif, metadata)
        
    elif metadata['current_model']['model_type']== 'RF':
        model = RF(metadata['current_model']['model_type'],metadata['current_model'])
        model.fit(df_train,df_verif,
                  metadata,
                  model_hyperparams_address=metadata['current_model']['model_hyperparam_address'],
                  sampling_type=metadata['current_model']['resampling_type'])
    elif metadata['current_model']['model_type']== 'Naive':
        model = Naive_Model(metadata['current_model']['model_type'],metadata['current_model'])
        model.fit(df_train,df_verif, metadata)    

    return model








    


