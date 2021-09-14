# config.conf
The user should carefully adjust the parameters in config.conf and feed that to the model.  For more information about each parameter please read config_readme.xlsx.




# rf_configs
The file rf.json contains hyperparameter and configuration information for training a random forest model. A typical user should avoid changing any parameters in this folder. The JSON object contains several important fields:

“search_type” – The value should be one of {“None”, “random_grid”, “full_grid”}, and defines the type of hyper-parameter search that will be carried out when training the model. If “None”, then no search will be made and the hyperparameters given in the “hyperparams” field (described below) will be used directly. If “random_grid” or “full_grid”, a search over the hyperparameter sets given in the “hyperparam_grids” field will be completed. “random_grid” will do a randomized search, while “full_grid” will do a complete grid search over all possible combinations of hyperparameters.

“class_weights” – this value should be one of {“None”, “balanced”, “balanced_subsample”}. This controls the weight given to each class during learning. “None” gives all classes weight one. “balanced” uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data, while “balanced_subsample” is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown (from sklearn documentation https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

“number_of_threads” – the number of threads to use when parallelizing the hyperparameter search. Shouldn’t exceed the number of virtual processors available on the computer.

“grid_n_iterations” – the number of hyperparameter combinations that are sampled if performing a random_grid search

“grid_scoring” – the metric to use when choosing the best model during a hyperparameter search, such as f1 score. Should be one of the scoring parameters defined by sklearn (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

“best_params_saved_loc” – the file name to use when saving the best hyperparameters found by the search.

“hyperparams” - This field is used to define model hyperparameters when not performing a search (i.e. “search_type” = “None”). It defines separate parameters for each cluster, in case different clusters have different optimal hyperparameter choices.

“hyperparam_grids” -  This field is used to define sets of hyperparameter values to explore for each cluster if performing a hyperparameter search (i.e. “search_type” = “random_grid” or “full_grid”).

