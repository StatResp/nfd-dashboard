"""
Base Class for all forecasters. All classes must implement the following methods:

1. fit -- fit a regression model to data
2. predict -- sample instances based on the forecasting model
3. get_regression_expr -- get a patsy expression for the regression.
4. update_model_stats -- store the model likelihood, AIC score for easy access
"""
import pickle
import os

class Forecaster:

    def __init__(self):
        self.model_params = None
        self.model_name = None
        self.model_stats = None

    def fit(self):
        pass

    def prediction(self):
        pass

    def get_regression_expr(self):
        pass

    def update_model_stats(self):
        pass

    def Likelihood(self):
        pass


    def save(self,Address):
        
        
        #directory='output/train/models/'
        directory=Address
        if not os.path.exists(directory):
            os.makedirs(directory)           
        if self.model_type !='NN' :               
            pickle.dump(self, open(directory+self.name+'.sav', 'wb'))
        else:
            self.saveNN(directory)
        print('Saving the model {} is done.'.format(self.name))
        
        
    def load(self,Address):
        # model = pickle.load(open('output/train/models/'+metadata['model_to_predict'], 'rb'))
        model= pickle.load(open(Address, 'rb'))
        if model.model_params[list(model.model_params.keys())[0]] is None:
            model=model.loadNN(Address)
        return model      
    
    
        