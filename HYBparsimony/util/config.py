#################################################
#****************LINEAR MODELS******************#
#################################################

CLASSIF_LOGISTIC_REGRESSION = {"C":{"range": (1., 100.), "type": 1}, 
                               "tol":{"range": (0.0001,0.9999), "type": 1}}
                            

CLASSIF_PERCEPTRON = {"tol":{"range": (0.0001,0.9999), "type": 1}, 
                      "alpha":{"range": (0.0001,0.9999), "type": 1}}


REG_LASSO = {"tol":{"range": (0.0001,0.9999), "type": 1}, 
             "alpha":{"range": (1., 100.), "type": 1}}


REG_RIDGE = {"tol":{"range": (0.0001,0.9999), "type": 1}, 
             "alpha":{"range": (1., 100.), "type": 1}}

################################################
#*****************SVM MODELS*******************#
################################################

CLASSIF_SVC = {"C":{"range": (1.,100.), "type": 1}, 
               "alpha":{"range": (0.0001,0.9999), "type": 1}}


REG_SVR = {"C":{"range": (1.,100.), "type": 1}, 
           "alpha":{"range": (0.0001,0.9999), "type": 1}}


##################################################
#******************KNN MODELS********************#
##################################################                            

CLASSIF_KNEIGHBORSCLASSIFIER = {"n_neighbors":{"range": (2,11), "type": 0}, 
                                "p":{"range": (1, 3), "type": 0}}


REG_KNEIGHBORSREGRESSOR = {"n_neighbors":{"range": (2,11), "type": 0}, 
                           "p":{"range": (1, 3), "type": 0}}


##################################################
#******************MLP MODELS********************#
##################################################                            

CLASSIF_MLPCLASSIFIER = {"tol":{"range": (0.0001,0.9999), "type": 1},
                        "alpha":{"range": (0.0001, 0.999), "type": 1}}


REG_MLPREGRESSOR = {"tol":{"range": (0.0001,0.9999), "type": 1},
                    "alpha":{"range": (0.0001, 0.999), "type": 1}}


##################################################
#*************Random Forest MODELS***************#
##################################################                            

CLASSIF_RANDOMFORESTCLASSIFIER = {"n_estimators":{"range": (100,250), "type": 0},
                        "max_depth":{"range": (4, 20), "type": 0},
                        "min_samples_split":{"range": (2,25), "type": 0}}


REG_RANDOMFORESTREGRESSOR = {"n_estimators":{"range": (100,250), "type": 0},
                        "max_depth":{"range": (4, 20), "type": 0},
                        "min_samples_split":{"range": (2,25), "type": 0}}


##################################################
#*************Decision trees MODELS**************#
##################################################                            

CLASSIF_DECISIONTREECLASSIFIER = {"min_weight_fraction_leaf":{"range": (0,20), "type": 0},
                        "max_depth":{"range": (4, 20), "type": 0},
                        "min_samples_split":{"range": (2,25), "type": 0}}


REG_DECISIONTREEREGRESSOR = {"min_weight_fraction_leaf":{"range": (0,20), "type": 0},
                        "max_depth":{"range": (4, 20), "type": 0},
                        "min_samples_split":{"range": (2,25), "type": 0}}