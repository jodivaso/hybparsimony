# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer
import numpy as np
import warnings
import os

def getFitness(algorithm, complexity, custom_eval_fun=cross_val_score, ignore_warnings = True):
    r"""
     Fitness function for hybparsimony.

    Parameters
    ----------
    algorithm : object
        The machine learning algorithm to optimize. 
    complexity : function
        A function that calculates the complexity of the model. There are some functions available in `hybparsimony.util.complexity`.
    custom_eval_fun : function
        An evaluation function similar to scikit-learns's 'cross_val_score()'  

    Returns
    -------
    float
        np.array([model's fitness value (J), model's complexity]), model

    Examples
    --------
    Usage example for a binary classification model 
    
    .. highlight:: python
    .. code-block:: python

        import pandas as pd
        import numpy as np
        from sklearn.datasets import load_breast_cancer
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from hybparsimony import hybparsimony
        from hybparsimony.util import getFitness, svm_complexity, population
        # load 'breast_cancer' dataset
        breast_cancer = load_breast_cancer()
        X, y = breast_cancer.data, breast_cancer.target
        chromosome = population.Chromosome(params = [1.0, 0.2],
                                        name_params = ['C','gamma'],
                                        const = {'kernel':'rbf'},
                                        cols= np.random.uniform(size=X.shape[1])>0.50,
                                        name_cols = breast_cancer.feature_names)
        print(getFitness(SVC,svm_complexity)(chromosome, X=X, y=y))
    """

    if algorithm is None:
        raise Exception("An algorithm function must be provided!!!")
    if complexity is None or not callable(complexity):
        raise Exception("A complexity function must be provided!!!")


    def fitness(cromosoma, **kwargs):
        if "pandas" in str(type(kwargs["X"])):
            kwargs["X"] = kwargs["X"].values
        if "pandas" in str(type(kwargs["y"])):
            kwargs["y"] = kwargs["y"].values

        X_train = kwargs["X"]
        y_train = kwargs["y"]
            
        try:
            # Extract features from the original DB plus response (last column)
            data_train_model = X_train[: , cromosoma.columns]

            if ignore_warnings:
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore"

            # train the model
            aux = algorithm(**cromosoma.params)
            fitness_val = custom_eval_fun(aux, data_train_model, y_train).mean()
            modelo = algorithm(**cromosoma.params).fit(data_train_model, y_train)

            # Reset warnings to default values
            warnings.simplefilter("default")
            os.environ["PYTHONWARNINGS"] = "default"

            # El híbrido funciona de forma que cuanto más alto es mejor. Por tanto, con RMSE deberíamos trabajar con su negación.
            return np.array([fitness_val, complexity(modelo, np.sum(cromosoma.columns))]), modelo
        except Exception as e:    
            print(e)
            return np.array([np.NINF, np.Inf]), None

    return fitness



def fitness_for_parallel(algorithm, complexity, custom_eval_fun=cross_val_score, cromosoma=None, 
                         X=None, y=None, ignore_warnings = True):
    r"""
     Fitness function for hybparsimony similar to 'getFitness()' without being nested, to allow the pickle and therefore the parallelism.

    Parameters
    ----------
    algorithm : object
        The machine learning algorithm to optimize. 
    complexity : function
        A function that calculates the complexity of the model. There are some functions available in `hybparsimony.util.complexity`.
    custom_eval_fun : function
        An evaluation function similar to scikit-learns's 'cross_val_score()'.
    cromosoma: population.Chromosome class
        Solution's chromosome.
    X : {array-like, dataframe} of shape (n_samples, n_features)
        Input matrix.
    y : {array-like, dataframe} of shape (n_samples,)
        Target values (class labels in classification, real numbers in regression).
    ignore_warnings: True
        If ignore warnings.

    Returns
    -------
    float
        np.array([model's fitness value (J), model's complexity]), model

    Examples
    --------

    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from hybparsimony import hybparsimony
    from hybparsimony.util import svm_complexity, population
    from hybparsimony.util.fitness import fitness_for_parallel
    # load 'breast_cancer' dataset
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    chromosome = population.Chromosome(params = [1.0, 0.2],
                                       name_params = ['C','gamma'],
                                       const = {'kernel':'rbf'},
                                       cols= np.random.uniform(size=X.shape[1])>0.50,
                                       name_cols = breast_cancer.feature_names)
    print(fitness_for_parallel(SVC, svm_complexity, 
                               custom_eval_fun=cross_val_score,
                               cromosoma=chromosome, X=X, y=y))

    """

    if "pandas" in str(type(X)):
        X = X.values
    if "pandas" in str(type(y)):
        y = y.values

    X_train = X
    y_train = y

    try:
        # Extract features from the original DB plus response (last column)
        data_train_model = X_train[:, cromosoma.columns]

        if ignore_warnings:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

        # train the model
        aux = algorithm(**cromosoma.params)
        fitness_val = custom_eval_fun(aux, data_train_model, y_train).mean()
        modelo = algorithm(**cromosoma.params).fit(data_train_model, y_train)

        # Reset warnings to default values
        warnings.simplefilter("default")
        os.environ["PYTHONWARNINGS"] = "default"

        # El híbrido funciona de forma que cuanto más alto es mejor. Por tanto, con RMSE deberíamos trabajar con negativos.

        return np.array([fitness_val, complexity(modelo, np.sum(cromosoma.columns))]), modelo
    except Exception as e:
        print(e)
        return np.array([np.NINF, np.Inf]), None