from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPRegressor
from HYBparsimony import Population
from HYBparsimony.util import linearModels_complexity, mlp_complexity
from HYBparsimony.util.complexity import kernel_ridge_complexity, generic_complexity
import warnings
import inspect

_valid_algorithms = ["Ridge", "KernelRidge", "MLPRegressor", "RidgeClassifier", "LogisticRegression"]
valid_algorithms = [x.lower() for x in _valid_algorithms]

def check_algorithm(algorithm):
    if algorithm is not None:
        if isinstance(algorithm, str):
            if algorithm.lower() not in valid_algorithms:
                raise ValueError(algorithm, "is not available")
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

#####################
# REGRESSION MODELS
# ###################

Ridge_Model = {"estimator": Ridge,
               "complexity": linearModels_complexity,
               "alpha": {"range": (-5, 3), "type": Population.POWER}
               }

KernelRidge_Model = {"estimator": KernelRidge,
                "complexity":kernel_ridge_complexity,
                "alpha": {"range": (-5, 3), "type": Population.POWER},
                "gamma": {"range": (-5, 3), "type": Population.POWER},
                "kernel": {"value": "rbf", "type": Population.CONSTANT}}

MLPRegressor_Model = {"estimator": MLPRegressor, # The estimator
                      "complexity": mlp_complexity, # The complexity
                      "hidden_layer_sizes": {"range": (1, 25), "type": Population.INTEGER},
                      "alpha": {"range": (-5, 3), "type": Population.POWER},
                      "solver": {"value": "lbfgs", "type": Population.CONSTANT},
                      "activation": {"value": "logistic", "type": Population.CONSTANT},
                      "n_iter_no_change": {"value": 20, "type": Population.CONSTANT},
                      "tol": {"value": 1e-5, "type": Population.CONSTANT},
                      "random_state": {"value": 1234, "type": Population.CONSTANT},
                      "max_iter": {"value": 5000, "type": Population.CONSTANT}}

#########################
# CLASSIFICATION MODELS
# #######################

RidgeClassifier_Model = {"estimator": RidgeClassifier,
               "complexity": linearModels_complexity,
               "alpha": {"range": (-5, 3), "type": Population.POWER}
               }

Logistic_Model = {"estimator": LogisticRegression,
                 "complexity": linearModels_complexity,
                 "C": {"range": (-5, 3), "type": Population.POWER}
                 }
