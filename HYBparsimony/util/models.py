from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from HYBparsimony import Population
from HYBparsimony.util.complexity import * 
import warnings
import inspect

# Function to return correct algorithm dictionary
def check_algorithm(algorithm, is_classification):
    if algorithm is not None:
        if isinstance(algorithm, str):
            if algorithm not in list(algorithms_dict.keys()):
                raise ValueError(algorithm, "is not available")
            else:
                algorithm = algorithms_dict[algorithm]
        elif isinstance(algorithm, dict):
            estimator = algorithm.get("estimator")
            if estimator is None:
                raise ValueError("The provided dictionary does not contain an estimator")
            elif not inspect.isclass(estimator): # Si no es una clase
                raise ValueError("The estimator is not a class")
            elif not hasattr(estimator,"fit"):
                raise ValueError("The estimator does not contain a fit method")
            if algorithm.get("complexity") is None:
                warnings.warn("The dictionary does not contain a complexity function, the default one (number of features) will be used")
                algorithm["complexity"] = generic_complexity
    else:
        algorithm=algorithms_dict['LogisticRegression'] if is_classification else algorithms_dict['Ridge'] 
    return algorithm   
            
algorithms_dict = dict(
    #####################
    # REGRESSION MODELS
    # ###################
    Ridge = {"estimator": Ridge,
               "complexity": linearModels_complexity,
               "alpha": {"range": (-5, 5), "type": Population.POWER}
               },

    Lasso = {"estimator": Lasso,
               "complexity": linearModels_complexity,
               "alpha": {"range": (-5, 5), "type": Population.POWER}
               },

    KernelRidge = {"estimator": KernelRidge,
                    "complexity":kernel_ridge_complexity,
                    "alpha": {"range": (-5, 5), "type": Population.POWER},
                    "gamma": {"range": (-5, 3), "type": Population.POWER},
                    "kernel": {"value": "rbf", "type": Population.CONSTANT}
                  },

    MLPRegressor = {"estimator": MLPRegressor, # The estimator
                  "complexity": mlp_complexity, # The complexity
                  "hidden_layer_sizes": {"range": (1, 25), "type": Population.INTEGER},
                  "alpha": {"range": (-5, 5), "type": Population.POWER},
                  "solver": {"value": "lbfgs", "type": Population.CONSTANT},
                  "activation": {"value": "logistic", "type": Population.CONSTANT},
                  "n_iter_no_change": {"value": 20, "type": Population.CONSTANT},
                  "tol": {"value": 1e-5, "type": Population.CONSTANT},
                  "random_state": {"value": 1234, "type": Population.CONSTANT},
                  "max_iter": {"value": 5000, "type": Population.CONSTANT}
                   },
    
    SVR = {"estimator": SVR,
                    "complexity":svm_complexity,
                    "C": {"range": (-5, 3), "type": Population.POWER},
                    "gamma": {"range": (-5, 0), "type": Population.POWER},
                    "kernel": {"value": "rbf", "type": Population.CONSTANT}
                  },

    DecisionTreeRegressor = {"estimator": DecisionTreeRegressor,
                    "complexity":decision_tree_complexity,
                    "max_depth":{"range": (2, 30), "type": Population.INTEGER},
                    "min_samples_split":{"range": (2,25), "type": Population.INTEGER},
                    # "min_weight_fraction_leaf":{"range": (0.0,0.50), "type": Population.FLOAT},
                  },

    RandomForestRegressor = {"estimator": RandomForestRegressor,
                    "complexity":randomForest_complexity,
                    "n_estimators":{"range": (10,500), "type": 0},
                    "max_depth":{"range": (2, 20), "type": Population.INTEGER},
                    "min_samples_split":{"range": (2,25), "type": Population.INTEGER},
                    # "min_weight_fraction_leaf":{"range": (0.0,0.50), "type": Population.FLOAT},
                  },

    KNeighborsRegressor = {"estimator": KNeighborsRegressor,
                    "complexity":knn_complexity,
                    "n_neighbors":{"range": (1,50), "type": Population.INTEGER}, 
                    "p":{"range": (1, 3), "type": Population.INTEGER},
                  },

    #########################
    # CLASSIFICATION MODELS
    # #######################
    RidgeClassifier = {"estimator": RidgeClassifier,
                   "complexity": linearModels_complexity,
                   "alpha": {"range": (-5, 3), "type": Population.POWER}
                   },

    LogisticRegression = {"estimator": LogisticRegression,
                     "complexity": linearModels_complexity,
                     "C": {"range": (-5, 3), "type": Population.POWER}
                     }
)
