import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from hybparsimony import HYBparsimony

if __name__ == "__main__":

    #####################################################
    #         Use sklearn regression algorithm          #
    #####################################################

    # Load 'diabetes' dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)
    
    # Standarize X and y (some algorithms require that)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    algo = 'KernelRidge'
    HYBparsimony_model = HYBparsimony(algorithm=algo,
                                    features=diabetes.feature_names,
                                    rerank_error=0.001,
                                    verbose=1)
    HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)
    # Check results with test dataset
    preds = HYBparsimony_model.predict(X_test)
    
    print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    print(f'Selected features:{HYBparsimony_model.selected_features}')
    print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    print(f'5-CV MSE = {-round(HYBparsimony_model.best_score,6)}')
    print(f'RMSE test = {round(mean_squared_error(y_test, preds, squared=False),6)}')
    
    

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
    #     HYBparsimony_model = hybparsimony(algorithm=algo,
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

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import log_loss
    from hybparsimony import HYBparsimony
    
    # load 'breast_cancer' dataset
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target 
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)
    
    # Standarize X and y (some algorithms require that)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

  
    HYBparsimony_model = HYBparsimony(features=breast_cancer.feature_names,
                                    #   cv=RepeatedKFold(n_splits=5, n_repeats=10),
                                      rerank_error=0.005,
                                      verbose=1)
    HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)
    # Extract probs of class==1
    preds = HYBparsimony_model.predict_proba(X_test)[:,1]
    print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    print(f'Selected features:{HYBparsimony_model.selected_features}')
    print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    print(f'5-CV logloss = {-round(HYBparsimony_model.best_score,6)}')
    print(f'logloss test = {round(log_loss(y_test, preds),6)}')



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
    #     HYBparsimony_model = hybparsimony(algorithm=algo,
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
    from hybparsimony import HYBparsimony
    
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
    HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)
    preds = HYBparsimony_model.predict(X_test)
    print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    print(f'Selected features:{HYBparsimony_model.selected_features}')
    print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    print(f'10R5-CV f1_macro = {round(HYBparsimony_model.best_score,6)}')
    print(f'f1_macro test = {round(f1_score(y_test, preds, average="macro"),6)}')
    

    ###################################################
    #                   CUSTOM EVALUATION             #
    ###################################################

    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.datasets import load_breast_cancer, load_wine
    # from sklearn.model_selection import cross_val_score
    # from hybparsimony import hybparsimony
    # from sklearn.metrics import fbeta_score, make_scorer, cohen_kappa_score, log_loss, accuracy_score
    

    # # load 'breast_cancer' dataset
    # breast_cancer = load_breast_cancer()
    # X, y = breast_cancer.data, breast_cancer.target 
    # print(X.shape)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)
    
    # # Standarize X and y (some algorithms require that)
    # scaler_X = StandardScaler()
    # X_train = scaler_X.fit_transform(X_train)
    # X_test = scaler_X.transform(X_test)

    # # #Example A: Using 10 folds and 'accuracy'
    # HYBparsimony_model = hybparsimony(features=breast_cancer.feature_names,
    #                                 scoring='accuracy',
    #                                 cv=10,
    #                                 rerank_error=0.001,
    #                                 verbose=1)
    
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.1)
    # preds = HYBparsimony_model.predict(X_test)
    # print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    # print(f'Selected features:{HYBparsimony_model.selected_features}')
    # print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    # print(f'10R5-CV Accuracy = {round(HYBparsimony_model.best_score,6)}')
    # print(f'Accuracy test = {round(accuracy_score(y_test, preds),6)}')


#     # load 'wine' dataset 
#     wine = load_wine()
#     X, y = wine.data, wine.target 
#     print(X.shape)
#     # 3 classes
#     print(len(np.unique(y)))

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)
    
#    # Standarize X and y (some algorithms require that)
#     scaler_X = StandardScaler()
#     X_train = scaler_X.fit_transform(X_train)
#     X_test = scaler_X.transform(X_test)

#     #Example B: Using 10-repeated 5-fold CV and 'Kappa' score
#     from sklearn.metrics import cohen_kappa_score, make_scorer
#     metric_kappa = make_scorer(cohen_kappa_score, greater_is_better=True)
#     HYBparsimony_model = hybparsimony(features=wine.feature_names,
#                                     scoring=metric_kappa,
#                                     cv=RepeatedKFold(n_splits=5, n_repeats=10),
#                                     rerank_error=0.001,
#                                     verbose=1)

#     #Example C: Using a weighted 'log_loss'
#     from sklearn.metrics import cohen_kappa_score, make_scorer
#     # Assign a double weight to class one
#     def my_custom_loss_func(y_true, y_pred):
#         sample_weight = np.ones_like(y_true)
#         sample_weight[y_true==1] = 2.0
#         return log_loss(y_true, y_pred, sample_weight=sample_weight)
#     # Lower is better and 'log_loss' needs probabilities
#     custom_score = make_scorer(my_custom_loss_func, greater_is_better=False, needs_proba=True)
#     HYBparsimony_model = hybparsimony(features=breast_cancer.feature_names,
#                                     scoring=custom_score,
#                                     rerank_error=0.001,
#                                     verbose=1)
    
#     # Example D: Using a 'custom evaluation' function
#     #          (Parallelism is not allowed)
#     def custom_fun(estimator, X, y):
#         return cross_val_score(estimator, X, y, scoring="accuracy")
    
#     HYBparsimony_model = hybparsimony(features=breast_cancer.feature_names,
#                                     custom_eval_fun=custom_fun,
#                                     n_jobs=1, #parallelism is not allowed with 'custom_eval_fun'
#                                     rerank_error=0.001,
#                                     verbose=1)


#     HYBparsimony_model.fit(X_train, y_train, time_limit=0.1)
#     preds = HYBparsimony_model.predict(X_test)
#     print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
#     print(f'Selected features:{HYBparsimony_model.selected_features}')
#     print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
#     print(f'10R5-CV Accuracy = {round(HYBparsimony_model.best_score,6)}')
#     print(f'Accuracy test = {round(accuracy_score(y_test, preds),6)}')

    
   
    # HYBparsimony_model = hybparsimony(n_jobs=1,custom_eval_fun=custom_fun) # Este con paralelismo NO funciona
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.5)
    # preds = HYBparsimony_model.predict(X_test)
    # print("Accuracy test", accuracy_score(y_test, preds))




    # ###################################################
    # #                   CUSTOM SEARCH                 #
    # ###################################################

    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    # from sklearn.neural_network import MLPRegressor
    # from sklearn.metrics import mean_squared_error
    # from sklearn.datasets import load_diabetes
    # from sklearn.preprocessing import StandardScaler
    # from hybparsimony import hybparsimony, Population

    # # Load 'diabetes' dataset
    # diabetes = load_diabetes()

    # X, y = diabetes.data, diabetes.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

    # # Standarize X and y
    # scaler_X = StandardScaler()
    # X_train = scaler_X.fit_transform(X_train)
    # X_test = scaler_X.transform(X_test)
    # scaler_y = StandardScaler()
    # y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
    # y_test = scaler_y.transform(y_test.reshape(-1,1)).flatten()

    # def mlp_new_complexity(model, nFeatures, **kwargs):
    #     weights = [np.concatenate(model.intercepts_)]
    #     for wm in model.coefs_:
    #         weights.append(wm.flatten())
    #     weights = np.concatenate(weights) 
    #     int_comp = np.min((1E09-1,np.sum(weights**2)))
    #     return nFeatures*1E09 + int_comp

    # MLPRegressor_new = {"estimator": MLPRegressor, # The estimator
    #               "complexity": mlp_new_complexity, # The complexity
    #               "hidden_layer_sizes": {"range": (1, 5), "type": Population.INTEGER},
    #               "alpha": {"range": (-5, 5), "type": Population.POWER},
    #               "solver": {"value": "adam", "type": Population.CONSTANT},
    #               "learning_rate": {"value": "adaptive", "type": Population.CONSTANT},
    #               "early_stopping": {"value": True, "type": Population.CONSTANT},
    #               "validation_fraction": {"value": 0.10, "type": Population.CONSTANT},
    #               "activation": {"value": "tanh", "type": Population.CONSTANT},
    #               "n_iter_no_change": {"value": 20, "type": Population.CONSTANT},
    #               "tol": {"value": 1e-5, "type": Population.CONSTANT},
    #               "random_state": {"value": 1234, "type": Population.CONSTANT},
    #               "max_iter": {"value": 200, "type": Population.CONSTANT}
    #                }
    # HYBparsimony_model = hybparsimony(algorithm=MLPRegressor_new,
    #                                 features=diabetes.feature_names,
    #                                 cv=RepeatedKFold(n_splits=5, n_repeats=10),
    #                                 npart = 10,
    #                                 rerank_error=0.001,
    #                                 verbose=1)

    # # Search the best hyperparameters and features 
    # # (increasing 'time_limit' to improve RMSE with high consuming algorithms)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=1.00)
    # preds = HYBparsimony_model.predict(X_test)
    # print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    # print(f'Selected features:{HYBparsimony_model.selected_features}')
    # print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    # print(f'5-CV MSE = {-round(HYBparsimony_model.best_score,6)}')
    # print(f'RMSE test = {round(mean_squared_error(y_test, preds, squared=False),6)}')
                    
    

    # ###################################################
    # # Check getFitness() and fitness_for_parallel()   #
    # ###################################################

    # import pandas as pd
    # import numpy as np
    # from sklearn.datasets import load_breast_cancer
    # from sklearn.svm import SVC
    # from sklearn.model_selection import cross_val_score
    # from hybparsimony import hybparsimony
    # from hybparsimony.util import getFitness, svm_complexity, population
    # from hybparsimony.util.fitness import fitness_for_parallel
    # # load 'breast_cancer' dataset
    # breast_cancer = load_breast_cancer()
    # X, y = breast_cancer.data, breast_cancer.target
    # chromosome = population.Chromosome(params = [1.0, 0.2],
    #                                    name_params = ['C','gamma'],
    #                                    const = {'kernel':'rbf'},
    #                                    cols= np.random.uniform(size=X.shape[1])>0.50,
    #                                    name_cols = breast_cancer.feature_names)
    # # print(getFitness(SVC,svm_complexity)(chromosome, X=X, y=y))
    # print(fitness_for_parallel(SVC, svm_complexity, 
    #                            custom_eval_fun=cross_val_score,
    #                            cromosoma=chromosome, X=X, y=y))




    # # #Example A: Using 10 folds and 'accuracy'
    # HYBparsimony_model = hybparsimony(features=breast_cancer.feature_names,
    #                                 scoring='accuracy',
    #                                 keep_history=True,
    #                                 cv=10,
    #                                 rerank_error=0.001,
    #                                 verbose=1)
    

    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.1)
    # preds = HYBparsimony_model.predict(X_test)
    # print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    # print(f'Selected features:{HYBparsimony_model.selected_features}')
    # print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    # print(f'10R5-CV Accuracy = {round(HYBparsimony_model.best_score,6)}')
    # print(f'Accuracy test = {round(accuracy_score(y_test, preds),6)}')
    


