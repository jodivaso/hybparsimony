# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer
import numpy as np
import warnings
import os

def getFitness(algorithm, metric, complexity, custom_eval_fun=cross_val_score, minimize=False, test_size=0.2, random_state=42, n_jobs=-1, ignore_warnings = True):
    r"""
     Fitness function for GAparsimony.

    Parameters
    ----------
    algorithm : object
        The machine learning function to optimize. 
    metric : function
        A function that computes the fitness value.
    complexity : function
        A function that calculates the complexity of the model. There are some functions available in `GAparsimony.util.complexity`.
    cv : object, optional
        An `sklearn.model_selection` function. By default, is defined `RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)`.
    minimize : bool, optional
        `False`, if the objective is to minimize the metric, to maximize it, set to `True`.
    test_size : float, int, None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, model is not tested with testing split returning fitness_test=np.inf. Default 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        Default `42`
    n_jobs : int, optional
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``-1`` means using all processors. Default `-1`
    Examples
    --------
    Usage example for a regression model 
    
    .. highlight:: python
    .. code-block:: python

        from sklearn.svm import SVC
        from sklearn.metrics import cohen_kappa_score

        from GAparsimony import getFitness
        from GAparsimony.util import svm_complexity

        fitness = getFitness(SVC, cohen_kappa_score, svm_complexity, cv, maximize=True, test_size=0.2, random_state=42, n_jobs=-1)
    """

    if algorithm is None:
        raise Exception("An algorithm function must be provided!!!")
    if metric is None or not callable(metric):
        raise Exception("A metric function must be provided!!!")
    if complexity is None or not callable(complexity):
        raise Exception("A complexity function must be provided!!!")


    def fitness(cromosoma, **kwargs):
        if "pandas" in str(type(kwargs["X"])):
            kwargs["X"] = kwargs["X"].values
        if "pandas" in str(type(kwargs["y"])):
            kwargs["y"] = kwargs["y"].values
        if test_size != None:
            X_train, X_test, y_train, y_test = train_test_split(kwargs["X"], kwargs["y"], test_size=test_size, random_state=random_state)
        else:
            X_train = kwargs["X"]
            y_train = kwargs["y"]
            
        try:
            # Extract features from the original DB plus response (last column)
            data_train_model = X_train[: , cromosoma.columns]
            if test_size != None:
                data_test_model = X_test[: , cromosoma.columns]

            if ignore_warnings:
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore"

            # train the model
            aux = algorithm(**cromosoma.params)
            #fitness_val = cross_val_score(aux, data_train_model, y_train, scoring=make_scorer(metric), cv=RepeatedKFold(n_splits=10, n_repeats=5), n_jobs=n_jobs).mean()
            fitness_val = custom_eval_fun(aux, data_train_model, y_train).mean()
            #print("f", fitness_val)
            modelo = algorithm(**cromosoma.params).fit(data_train_model, y_train)
            if test_size != None:
                fitness_test = metric(modelo.predict(data_test_model), y_test)
            else:
                fitness_test = np.inf

            # Reset warnings to default values
            warnings.simplefilter("default")
            os.environ["PYTHONWARNINGS"] = "default"

            # El híbrido funciona de forma que cuanto más alto es mejor. Por tanto, con RMSE deberíamos trabajar con su negación.
            if minimize:
                fitness_val = -fitness_val
                if test_size != None:
                    fitness_test = -fitness_test

            return np.array([fitness_val, fitness_test, complexity(modelo, np.sum(cromosoma.columns))]), modelo
        except Exception as e:    
            print(e)
            return np.array([np.NINF, np.NINF, np.Inf]), None

    return fitness
