import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from HYBparsimony import HYBparsimony

if __name__ == "__main__":

    # #####################################################
    # #         Use sklearn regression algorithm          #
    # #####################################################

    # Load 'diabetes' dataset
    # diabetes = load_diabetes()
    # X, y = diabetes.data, diabetes.target
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)
    
    # # Standarize X and y (some algorithms require that)
    # scaler_X = StandardScaler()
    # X_train = scaler_X.fit_transform(X_train)
    # X_test = scaler_X.transform(X_test)

    # scaler_y = StandardScaler()
    # y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
    # y_test = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    # algo = 'KernelRidge'
    # HYBparsimony_model = HYBparsimony(algorithm=algo,
    #                                 features=diabetes.feature_names,
    #                                 rerank_error=0.001,
    #                                 verbose=1)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)
    # # Check results with test dataset
    # preds = HYBparsimony_model.predict(X_test)
    
    # print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    # print(f'Selected features:{HYBparsimony_model.selected_features}')
    # print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    # print(f'5-CV MSE = {-round(HYBparsimony_model.best_score,6)}')
    # print(f'RMSE test = {round(mean_squared_error(y_test, preds, squared=False),6)}')
    
    

    ########################################################
    #         Use different regression algorithms          #
    ########################################################
    # algorithms_reg = ['Ridge', 'Lasso', 
    #                 'KernelRidge', 'KNeighborsRegressor',
    #                 'MLPRegressor', 'SVR',
    #                 'DecisionTreeRegressor', 'RandomForestRegressor'
    #                 ]
    # res = []
    # for algo in algorithms_reg:
    #     print('#######################')
    #     print('Searching best: ', algo)
    #     HYBparsimony_model = HYBparsimony(algorithm=algo,
    #                                       features=diabetes.feature_names,
    #                                       rerank_error=0.001,
    #                                       cv=RepeatedKFold(n_splits=5, n_repeats=10),
    #                                       maxiter=1000,
    #                                       verbose=1)
    #     # Search the best hyperparameters and features 
    #     # (increasing 'time_limit' to improve RMSE with high consuming algorithms)
    #     HYBparsimony_model.fit(X_train, y_train, time_limit=60)
    #     # Check results with test dataset
    #     preds = HYBparsimony_model.predict(X_test)
    #     print(algo, "RMSE test", mean_squared_error(y_test, preds, squared=False))
    #     print('Selected features:',HYBparsimony_model.selected_features)
    #     print(HYBparsimony_model.best_model)
    #     print('#######################')
    #     # Append results
    #     res.append(dict(algo=algo,
    #                     MSE_5CV= -round(HYBparsimony_model.best_score,6),
    #                     RMSE=round(mean_squared_error(y_test, preds, squared=False),6),
    #                     NFS=int(HYBparsimony_model.best_complexity//1e9),
    #                     selected_features = HYBparsimony_model.selected_features,
    #                     best_model=HYBparsimony_model.best_model))
    # res = pd.DataFrame(res).sort_values('RMSE')
    # res.to_csv('res_models.csv')
    # # Visualize results
    # print(res[['algo', 'MSE_5CV', 'RMSE', 'NFS']])



    # ##############################################################
    # #                      BINARY CLASSIFICATION                 #
    # ##############################################################

    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.datasets import load_breast_cancer
    # from sklearn.metrics import log_loss
    # from HYBparsimony import HYBparsimony
    
    # # load 'breast_cancer' dataset
    # breast_cancer = load_breast_cancer()
    # X, y = breast_cancer.data, breast_cancer.target 
    # print(X.shape)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)
    
    # # Standarize X and y (some algorithms require that)
    # scaler_X = StandardScaler()
    # X_train = scaler_X.fit_transform(X_train)
    # X_test = scaler_X.transform(X_test)

  
    # HYBparsimony_model = HYBparsimony(features=breast_cancer.feature_names,
    #                                 #   cv=RepeatedKFold(n_splits=5, n_repeats=10),
    #                                   rerank_error=0.005,
    #                                   verbose=1)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.50)
    # # Extract probs of class==1
    # preds = HYBparsimony_model.predict_proba(X_test)[:,1]
    # print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    # print(f'Selected features:{HYBparsimony_model.selected_features}')
    # print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    # print(f'5-CV logloss = {-round(HYBparsimony_model.best_score,6)}')
    # print(f'logloss test = {round(log_loss(y_test, preds),6)}')



    ###########################################################
    #         Use different classification algorithms         #
    ###########################################################
    # algorithms_clas = ['LogisticRegression', 'MLPClassifier', 
    #                 'SVC', 'DecisionTreeClassifier',
    #                 'RandomForestClassifier', 'KNeighborsClassifier',
    #                 ]
    # res = []
    # for algo in algorithms_clas:
    #     print('#######################')
    #     print('Searching best: ', algo)
    #     HYBparsimony_model = HYBparsimony(algorithm=algo,
    #                                       features=breast_cancer.feature_names,
    #                                       rerank_error=0.005,
    #                                       cv=RepeatedKFold(n_splits=5, n_repeats=10),
    #                                       maxiter=1000,
    #                                       verbose=1)
    #     # Search the best hyperparameters and features 
    #     # (increasing 'time_limit' to improve neg_log_loss with high consuming algorithms)
    #     HYBparsimony_model.fit(X_train, y_train, time_limit=60.0)
    #     # Check results with test dataset
    #     preds = HYBparsimony_model.predict_proba(X_test)[:,1]
    #     print(algo, "Logloss_Test=", round(log_loss(y_test, preds),6))
    #     print('Selected features:',HYBparsimony_model.selected_features)
    #     print(HYBparsimony_model.best_model)
    #     print('#######################')
    #     # Append results
    #     res.append(dict(algo=algo,
    #                     Logloss_10R5CV= -round(HYBparsimony_model.best_score,6),
    #                     Logloss_Test = round(log_loss(y_test, preds),6),
    #                     NFS=int(HYBparsimony_model.best_complexity//1e9),
    #                     selected_features = HYBparsimony_model.selected_features,
    #                     best_model=HYBparsimony_model.best_model))
    # res = pd.DataFrame(res).sort_values('Logloss_Test')
    # res.to_csv('res_models_class.csv')
    # # Visualize results
    # print(res[['algo', 'Logloss_10R5CV', 'Logloss_Test', 'NFS']])




    ###############################################################
    #                   MULTICLASS CLASSIFICATION                 #
    ###############################################################

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_wine
    from sklearn.metrics import f1_score
    from HYBparsimony import HYBparsimony
    
    # load 'wine' dataset 
    wine = load_wine()
    X, y = wine.data, wine.target 
    print(X.shape)
    # 3 classes
    print(len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)
    
    # Standarize X and y (some algorithms require that)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    HYBparsimony_model = HYBparsimony(features=wine.feature_names,
                                    cv=RepeatedKFold(n_splits=5, n_repeats=10),
                                    npart = 20, # population=20
                                    early_stop=20,
                                    rerank_error=0.001,
                                    verbose=1)
    HYBparsimony_model.fit(X_train, y_train, time_limit=5.0)
    preds = HYBparsimony_model.predict(X_test)
    print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    print(f'Selected features:{HYBparsimony_model.selected_features}')
    print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    print(f'10R5-CV f1_macro = {round(HYBparsimony_model.best_score,6)}')
    print(f'f1_macro test = {round(f1_score(y_test, preds, average="macro"),6)}')
    
