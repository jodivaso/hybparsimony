from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from HYBparsimony import Population, HYBparsimony

#Y si le paso una custom, ¿Cómo sé si hay que minimizar? Realmente tengo un problema y es que tengo que pasar métrica!

if __name__ == "__main__":

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


    ###############################################################
    #                     EJEMPLO CUSTOM_EVAL                     #
    ###############################################################

    # def custom_fun(estimator, X, y): # CV con repeatedKfold.
    #     return cross_val_score(estimator, X, y, cv=RepeatedKFold(n_splits=10, n_repeats=5))
    #
    # HYBparsimony_model = HYBparsimony(custom_eval_fun=custom_fun)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_train)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                     EJEMPLO OTRO ALGORITMO                  #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(algorithm="MLPRegressor")
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_train)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                     EJEMPLO FEATURES PARAMS                 #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(features = diabetes.feature_names)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_train)
    # print("RMSE test", mean_squared_error(y_test, preds))


    # f = make_scorer(mean_squared_error)
    # preds = Ridge().fit(X,y).predict(X)
    # print(f._score_func(preds,y)) #Esto es una ñapa tremenda! Del Scorer accedo al score_func original.






