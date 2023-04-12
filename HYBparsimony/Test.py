from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_diabetes

from HYBparsimony import Population, getFitness, HYBparsimony
from HYBparsimony.util import linearModels_complexity, models
from sklearn.linear_model import Ridge

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X = StandardScaler().fit_transform(X)

params = {"alpha":{"range": (1., 25.9), "type": Population.FLOAT},
            "tol":{"range": (0.0001,0.9999), "type": Population.FLOAT}}

# params = {"alpha":{"range": (-3, 5), "type": Population.POWER}}

rerank_error=0.01

fitness = getFitness(Lasso, mean_squared_error, linearModels_complexity, minimize=True, test_size=0.2, random_state=42, n_jobs=-1)

# HYBparsimony_model = HYBparsimony(fitness=fitness,
#                                 params = params,
#                                 features = diabetes.feature_names,
#                                 keep_history = True,
#                                 rerank_error = rerank_error,
#                                 npart=5,
#                                 maxiter=3)
#
# HYBparsimony_model.fit(X, y)

HYBparsimony_model = HYBparsimony(features=diabetes.feature_names, npart=5, maxiter=3)
HYBparsimony_model.fit(X, y)