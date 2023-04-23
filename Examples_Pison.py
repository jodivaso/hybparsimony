import numpy as np
from sklearn.linear_model import Lasso, RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, mean_absolute_error
from sklearn.datasets import load_diabetes, load_iris
from sklearn.neighbors import KNeighborsRegressor
from HYBparsimony import Population, HYBparsimony, order
from HYBparsimony.hybparsimony import default_cv_score_classification
from HYBparsimony.util import knn_complexity

if __name__ == "__main__":

    # REGRESIÓN
    # Cargo un dataset de regresión
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    ###############################################################
    #                       EJEMPLO BÁSICO                        #
    ###############################################################
    HYBparsimony_model = HYBparsimony()
    HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    preds = HYBparsimony_model.predict(X_test)
    print("RMSE test", mean_squared_error(y_test, preds))
    print('15 particulas')

    ###############################################################
    #                       EJEMPLO OTRO SCORING                  #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(scoring="neg_mean_absolute_error")
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("MAE test", mean_absolute_error(y_test, preds))

    ###############################################################
    #                          EJEMPLO OTRO CV                    #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(cv=RepeatedKFold(n_splits=10, n_repeats=5))
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                EJEMPLO OTRO SCORING Y OTRO CV               #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(scoring="neg_mean_absolute_error", cv=RepeatedKFold(n_splits=10, n_repeats=5))
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("MAE test", mean_absolute_error(y_test, preds))


    ###############################################################
    #                     EJEMPLO CUSTOM_EVAL                     #
    ###############################################################

    # def custom_fun(estimator, X, y): # CV con repeatedKfold.
    #     return cross_val_score(estimator, X, y, cv=RepeatedKFold(n_splits=10, n_repeats=5))
    
    # # Este con paralelismo NO funciona (sí funciona si el custom_fun lo definimos fuera del if main)
    # HYBparsimony_model = HYBparsimony(n_jobs=1, custom_eval_fun=custom_fun)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                     EJEMPLO OTRO ALGORITMO                  #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(algorithm="MLPRegressor")
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                     EJEMPLO FEATURES PARAMS                 #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(features = diabetes.feature_names)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))


    ###############################################################
    #                     EJEMPLO NUEVO ALGORITMO                 #
    ###############################################################

    # KNN_Model = {"estimator": KNeighborsRegressor,
    #              "complexity": knn_complexity,
    #              "n_neighbors": {"range": (1, 10), "type": Population.INTEGER}
    #              }
    # HYBparsimony_model = HYBparsimony(algorithm=KNN_Model)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # print(HYBparsimony_model.best_model)
    # print(HYBparsimony_model.selected_features)


    # CLASIFICACIÓN

    # Cargo un dataset de clasificación
    # iris = load_iris()
    # X, y = iris.data, iris.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=12345)

    ###############################################################
    #                       EJEMPLO BÁSICO                        #
    ###############################################################

    # HYBparsimony_model = HYBparsimony()
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict_proba(X_test)
    # print("Log loss test", log_loss(y_test, preds))

    ###############################################################
    #                     EJEMPLO CUSTOM_EVAL                     #
    ###############################################################

    # def custom_fun(estimator, X, y):
    #     return cross_val_score(estimator, X, y, scoring="accuracy")
    
    # HYBparsimony_model = HYBparsimony(n_jobs=1,custom_eval_fun=custom_fun) # Este con paralelismo NO funciona
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.5)
    # preds = HYBparsimony_model.predict(X_test)
    # print("Accuracy test", accuracy_score(y_test, preds))


    ###############################################################
    #                        EJEMPLO OTRO CV
    ###############################################################

    # HYBparsimony_model = HYBparsimony(cv=RepeatedKFold(n_splits=10, n_repeats=5))
    # HYBparsimony_model.fit(X_train, y_train, time_limit=1)
    # preds = HYBparsimony_model.predict(X_test)
    # print("Accuracy test", accuracy_score(y_test, preds))




