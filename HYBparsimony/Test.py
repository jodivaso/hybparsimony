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

#Y si le paso una custom, ¿Cómo sé si hay que minimizar? Realmente tengo un problema y es que tengo que pasar métrica!
if __name__ == "__main__":

    HYBparsimony_model = HYBparsimony()
    HYBparsimony_model.fit(X, y, time_limit=0.25)
    #preds = HYBparsimony_model.best_model.predict(X) # No hay predict??? Hacerlo! Y que también el fit lo devuelva.

    # f = make_scorer(mean_squared_error)
    # preds = Ridge().fit(X,y).predict(X)
    # print(f._score_func(preds,y)) #Esto es una ñapa tremenda! Del Scorer accedo al score_func original.






