from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, make_scorer

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

# El CV y el scoring debería meterselo al custom_fun.

def custom_fun (estimator, X, y):
    return cross_val_score(estimator,X,y, cv=RepeatedKFold(n_splits=10, n_repeats=5))

if __name__ == '__main__':
    HYBparsimony_model = HYBparsimony()
    HYBparsimony_model.fit(X, y, time_limit=0.25)





