from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from HYBparsimony import Population
from HYBparsimony.util import linearModels_complexity, mlp_complexity
from HYBparsimony.util.complexity import kernel_ridge_complexity

Ridge_Model = {"estimator": Ridge,
               "complexity": linearModels_complexity,
               "alpha": {"range": (-5, 3), "type": Population.POWER}
               }


KRidge_Model = {"estimator": KernelRidge,
                "complexity":kernel_ridge_complexity,
                "alpha": {"range": (0, 1), "type": Population.FLOAT},
                "gamma": {"range": (0, 1), "type": Population.FLOAT},
                "kernel": {"value": "rbf", "type": Population.CONSTANT}}

MLPRegressor_Model = {"estimator": MLPRegressor, # The estimator
                      "complexity": mlp_complexity, # The complexity
                      "hidden_layer_sizes": {"range": (1, 25), "type": Population.INTEGER},
                      "alpha": {"range": (0, 1), "type": Population.FLOAT},
                      "solver": {"value": "lbfgs", "type": Population.CONSTANT},
                      "activation": {"value": "logistic", "type": Population.CONSTANT},
                      "n_iter_no_change": {"value": 20, "type": Population.CONSTANT},
                      "tol": {"value": 1e-5, "type": Population.CONSTANT},
                      "random_state": {"value": 1234, "type": Population.CONSTANT},
                      "max_iter": {"value": 5000, "type": Population.CONSTANT}}