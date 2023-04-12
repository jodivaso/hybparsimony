# -*- coding: utf-8 -*-

"""Complexity module.

This module contains predefined complexity functions for some of the most popular algorithms in the scikit-learn library:

* **linearModels_complexity**: Any algorithm from `sklearn.linear_model'. Returns: 10^9·nFeatures + (sum of the squared coefs).
* **svm_complexity**: Any algorithm from `sklearn.svm'. Returns: 10^9·nFeatures + (number of support vectors).
* **knn_complexity**: Any algorithm from `sklearn.neighbors'. Returns: 10^9·nFeatures + 1/(number of neighbors)
* **mlp_complexity**: Any algorithm from `sklearn.neural_network'. Returns: 10^9·nFeatures + (sum of the ANN squared weights).
* **randomForest_complexity**: Any algorithm from `sklearn.ensemble.RandomForestRegressor' or 'sklearn.ensemble.RandomForestClassifier'. Returns: 10^9·nFeatures + (the average of tree leaves).
* **xgboost_complexity**: XGboost sklearn model. Returns: 10^9·nFeatures + (the average of tree leaves * number of trees) (Experimental)
* **decision_tree_complexity**: Any algorithm from 'sklearn.tree'. Return: 10^9·nFeatures + (number of leaves) (Experimental)

Otherwise:

* **generic_complexity**: Any algorithm. Returns: the number of input features (nFeatures).

Other complexity functions can be defined with the following interface.

.. highlight:: python
.. code-block:: python

    def complexity(model, nFeatures, **kwargs):
        pass
            
    return complexity
"""

import numpy as np

def generic_complexity(model, nFeatures, **kwargs):
    r"""
    Generic complexity function.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of input features the model has been trained with.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        nFeatures.

    """
    return nFeatures

def linearModels_complexity(model, nFeatures, **kwargs):
    r"""
    Complexity function for linear models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
         The number of input features the model has been trained with.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        10^9·nFeatures + (sum of the model squared coefs).

    """

    int_comp = np.min((1E09-1,np.sum(model.coef_**2))) # Internal Complexity Sum of squared weigths
    return nFeatures*1E09 + int_comp

def svm_complexity(model, nFeatures, **kwargs):
    r"""
    Complexity function for SVM models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
         The number of input features the model has been trained with.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        10^9·nFeatures + (number of support vectors)

    """

    int_comp = np.min((1E09-1,np.sum(model.n_support_))) # Internal Complexity
    return nFeatures*1E09 + int_comp

def knn_complexity(model, nFeatures, **kwargs):
    r"""
    Complexity function for KNN models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
         The number of input features the model has been trained with.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        10^9·nFeatures + 1/(number of neighbors)

    """

    int_comp = 1E06 * 1/model.n_neighbors   # More k less flexible
    return nFeatures*1E09 + int_comp

def mlp_complexity(model, nFeatures, **kwargs):
    r"""
    Complexity function for MLP models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
         The number of input features the model has been trained with.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        10^9·nFeatures + (sum of the ANN squared weights)

    """

    weights = [np.concatenate(model.intercepts_)]
    for wm in model.coefs_:
        weights.append(wm.flatten())
    weights = np.concatenate(weights) 
    int_comp = np.min((1E09-1,np.sum(weights**2)))
    return nFeatures*1E09 + int_comp

def randomForest_complexity(model, nFeatures, **kwargs):
    r"""
    Complexity function for Random Forest models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of input features the model has been trained with.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        10^9·nFeatures + (the average of tree leaves)

    """

    num_leaves = [tree.get_n_leaves() for tree in model.estimators_]
    int_comp = np.min((1E09-1,np.mean(num_leaves))) # More leaves more complex  
    return nFeatures*1E09 + int_comp

def xgboost_complexity(model, nFeatures, **kwargs):
    r"""
    Complexity function for XGBoost model.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
         The number of input features the model has been trained with.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        10^9·nFeatures + (the average of tree leaves * number of trees) (Experimental)

    """
    df_model = model.get_booster().trees_to_dataframe()
    df_model = df_model[df_model.Feature=='Leaf']
    mean_leaves = df_model.groupby('Tree')['Feature'].count().mean()
    num_trees = df_model.Tree.nunique()   
    int_comp = np.min((1E09-1,num_trees*mean_leaves))
    return nFeatures*1E09 + int_comp


def decision_tree_complexity(model, nFeatures, **kwargs):
    r"""
    Complexity function for decision tree model.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
         The number of input features the model has been trained with.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        10^9·nFeatures + (number of leaves)

    """
    num_leaves = model.get_n_leaves()
    int_comp = np.min((1E09-1,num_leaves)) # More leaves more complex  
    return nFeatures*1E09 + int_comp

