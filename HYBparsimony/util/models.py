from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from HYBparsimony import Population
from HYBparsimony.util import linearModels_complexity

Ridge_Model = {"estimator" : Ridge,
               "alpha": {"range": (0, 3), "type": Population.FLOAT},
               "complexity" : linearModels_complexity
               }