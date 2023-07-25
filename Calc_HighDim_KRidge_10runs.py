import copy
import os
import pandas as pd
import numpy as np
import math
import glob, random, time, sys
from string import printable
from datetime import datetime

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import RepeatedKFold, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, KFold
from sklearn.metrics import make_scorer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SequentialFeatureSelector

from hybparsimony import Population, HYBparsimony, order
from hybparsimony.hybparsimony import default_cv_score_classification
from hybparsimony.util import knn_complexity

# from PSOparsimony_Nested import PSOparsimony
from bayes_opt import BayesianOptimization
# from hybparsimony import hybparsimony
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import __version__ as sk_version

DIR_SALIDA = '../experiments_27abr2023/'
NUM_RUNS = 25
PBOUNDS = dict(alpha=(-5, 5), gamma=(-7,0))

tiempos = [['slice_norm_reduc.csv', 3, 2000, 378, 0.50], 
           ['blog_norm.csv', 3, 2000, 276, 0.50], 
           ['crime_norm.csv', 3, 2000, 127, 0.50], 
           ['tecator_norm.csv', 3, 2000, 124, 0.50],
           ['ailerons_norm.csv', 3, 2000, 40, 0.50], 
           ['puma_norm.csv', 3, 2000, 32, 0.50],
           ['bank_norm.csv', 3, 2000, 32, 0.50], 
           ['pol_norm.csv', 3, 2000, 26, 0.50],
           ['cpu_act_norm.csv', 3, 2000, 21, 0.50],
           ['elevators_norm.csv', 3, 2000, 18, 0.50],
           ['meta_norm.csv', 3, 2000, 17, 0.50],
           ['bodyfat_norm.csv', 3, 2000, 14, 0.50], 
           ['boston_norm.csv', 3, 2000, 13, 0.50],
           ['housing_norm.csv', 3, 2000, 13, 0.50],
           ['concrete_norm.csv', 3, 2000, 8, 0.50],
           ['no2_norm.csv', 3, 2000, 7, 0.50],
           ['pm10_norm.csv', 3, 2000, 7, 0.50],
           ['strike_norm.csv', 3, 2000, 6, 0.50]
          ]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# Optimize with BayesianOptimization
def optimize_fun_bayes(alpha, gamma):
    global X_train_val_bayes, y_train_val
    model = KernelRidge(alpha=10.0**alpha, gamma=10.0**gamma, kernel='rbf')
    return -np.sqrt(-np.mean(cross_val_score(model, X_train_val_bayes, y_train_val, cv=5, scoring="neg_mean_squared_error")))

num_run = 0
def custom_fun(estimator, Xcus, ycus):
    global num_run
    # return cross_val_score(estimator, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=num_run), scoring="neg_mean_squared_error")
    x_train_custom, x_test_custom, y_train_custom, y_test_custom = train_test_split(Xcus, ycus, test_size=0.20, shuffle=True, random_state=num_run)
    estimator.fit(x_train_custom, y_train_custom)
    return(-mean_squared_error(y_test_custom, estimator.predict(x_test_custom),squared=False))
    # return cross_val_score(estimator, X, y, cv=, scoring="neg_mean_squared_error")


if __name__ == "__main__":
    print(sk_version, pd.__version__, np.__version__)
    now = datetime.now()
    NAME_FILE = 'hyb_all_dbs' + now.strftime("%Y_%m_%d_%H_%M_%S_") 


    if not os.path.exists(DIR_SALIDA):
        os.mkdir(DIR_SALIDA)
    
    # Datasets loop
    out_res = []
    res_bayes = []
    for run_total in range(10):
        for names_tiempo in tiempos:
            ini_time = time.time()
            name_db = names_tiempo[0].split('_')[0]
            num_cols = names_tiempo[3]
            gamma_crossover = names_tiempo[4]
            list_csv = os.listdir('../datasets')
            list_csv = [i for i in list_csv if f'{name_db}' in i]
            file_db = list_csv[0]
            time_limit = int(names_tiempo[1])  # Time limit in minutes
            
            # Train with a max of 2000 rows
            df_VAL = pd.read_csv('../datasets/' + file_db)
            train_val_size = df_VAL.shape[0]//2
            if train_val_size>2000:
                train_val_size = 2000
            test_size = df_VAL.shape[0] - train_val_size
            names_tiempo[2] = train_val_size
            print(f'run_total={run_total} Hyb gamma_crossover={gamma_crossover} name_db={name_db} time_limit={time_limit} num_cols={num_cols} train_val_size={train_val_size} test_size={test_size}')
    
            scaler_pruebas = StandardScaler()
            VAL_norm = pd.DataFrame(scaler_pruebas.fit_transform(df_VAL), columns=df_VAL.columns)


            input_names, target_name = VAL_norm.columns[:-1], VAL_norm.columns[-1]
            X, y = VAL_norm[input_names], VAL_norm[target_name]

            X_train_val, X_test, y_train_val, y_test = train_test_split(X.values, y.values, 
                                                                        train_size = names_tiempo[2] / len(X), 
                                                                        random_state=12345)

            # # Fordward selecion
            # res_forward = []
            # cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2020)
            # for numf in np.arange(len(input_names)-1):
            #     model = Ridge(alpha=215.05373555710105)
            #     clf_backward = SequentialFeatureSelector(model, n_features_to_select=numf+1, direction="forward", 
            #                                         cv=cv, scoring='neg_mean_squared_error', n_jobs=-1).fit(X_train_val.values, y_train_val.values)
            #     final_feats = np.array(input_names)[clf_backward.get_support()]
            #     X_train_val_bayes = X_train_val[final_feats].copy()
            #     model.fit(X_train_val_bayes.values, y_train_val.values)
            #     preds = model.predict(X_test[final_feats].values)
            #     RMSE_TST = mean_squared_error(y_test, preds, squared=False)
            #     RMSE_VAL = -np.mean(cross_val_score(model, X_train_val_bayes.values, y_train_val.values, cv=cv, scoring="neg_mean_squared_error"))
                        
            #     print(f'Numf={numf} RMSE_VAL={RMSE_VAL:.4f} RMSE_TST={RMSE_TST:.4f} -> {final_feats}')
            #     res_forward.append(dict(numf=numf, RMSE_VAL=RMSE_VAL, RMSE_TST=RMSE_TST, final_feats=final_feats))
            # res_forward = pd.DataFrame(res_forward)
            # res_forward.iloc[:,:4]
            # res_forward.to_csv(DIR_SALIDA + 'fordward_' + name_db + '__' + NAME_FILE + '.csv', index=False)
            # # -----------

            # Main Loop
            probs_feats = []
            for num_run in range(NUM_RUNS):
                seed_everything(num_run+run_total*100)
                HYBparsimony_model = HYBparsimony(algorithm='KernelRidge',
                                                features=input_names,
                                                rerank_error=0.001,
                                                custom_eval_fun=custom_fun,
                                                gamma_crossover=gamma_crossover,
                                                seed_ini=num_run+run_total*100,
                                                npart = 15,
                                                maxiter=200,
                                                early_stop=35,
                                                verbose=1,
                                                n_jobs=1)
                HYBparsimony_model.fit(X_train_val, y_train_val, time_limit=8.0)

                best_model_parameters = HYBparsimony_model.best_model_conf[:-len(input_names)]
                best_model_probsfeats = HYBparsimony_model.best_model_conf[-len(input_names):]
                selec_feats_thr = list(input_names[best_model_probsfeats>=0.50])
                assert all(HYBparsimony_model.selected_features==selec_feats_thr)
                selec_bool = best_model_probsfeats>=0.50
                NFS = len(selec_feats_thr)
                print(best_model_parameters[0])
                model = KernelRidge(alpha=10.0**best_model_parameters[0], gamma=10.0**best_model_parameters[1], kernel='rbf')
                model.fit(X_train_val[:,selec_bool], y_train_val)
                preds = model.predict(X_test[:,selec_bool])
                
                # X_train_val_bayes = X_train_val[selec_feats_thr]
                # print('CV=', optimize_fun_bayes(alpha=best_model_parameters[0]))
                # pd.DataFrame(HYBparsimony_model.bestSolList).to_csv('../experiments_23abr2023/solutions.csv', index=False)
                # print(HYBparsimony_model.best_model_conf)
                # HYBparsimony_model.fit(X_train_val.values, y_train_val.values, time_limit=1)
                # preds = HYBparsimony_model.predict(X_test.values)
                RMSE_val = -custom_fun(model, X_train_val[:,selec_bool], y_train_val)
                RMSE_test = mean_squared_error(y_test, preds, squared=False)
                
                model = KernelRidge(alpha=10.0**best_model_parameters[0], gamma=10.0**best_model_parameters[1], kernel='rbf')
                model.fit(X_train_val, y_train_val)
                preds = model.predict(X_test)
                RMSE_test_all = mean_squared_error(y_test, preds, squared=False)
                print(f'{NAME_FILE} name_db={name_db} run={num_run} NFS={NFS} RMSE_val={RMSE_val} RMSE_test={RMSE_test} RMSE_test_all={RMSE_test_all}')
                print(HYBparsimony_model.best_model)

                out_res.append(dict(name_db=name_db, run_total=run_total, num_run=num_run,
                                    time_limit=time_limit, 
                                    num_cols=num_cols, train_val_size=train_val_size, test_size=test_size,
                                    NFS = NFS,
                                    RMSE_val = RMSE_val,
                                    RMSE=RMSE_test,
                                    best_model_parameters=best_model_parameters,
                                    best_model_probsfeats=best_model_probsfeats))
                probs_feats.append(best_model_probsfeats)
            probs_feats = pd.DataFrame(probs_feats, columns=input_names)
            probs_feats.to_csv(DIR_SALIDA + 'probs_' + name_db + '__' + NAME_FILE + '.csv', index=False)
            out_res_df = pd.DataFrame(out_res) 

            print(out_res_df.head())
            out_res_df.to_csv(DIR_SALIDA + 'out_res_' + NAME_FILE + '.csv', index=False)

            # probs_feats = pd.read_csv('../experiments_23abr2023/probs_slice__hyb_all_dbs2023_04_26_09_23_50_.csv')

            # Bayesian Optimization
            # ---------------------
            probs_feats_mean = probs_feats.mean(axis=0).values
            for thr_features in np.arange(0.00, 0.90, 0.025):
                selec_feats_thr = input_names[probs_feats_mean>=thr_features]
                selec_bool = probs_feats_mean>=thr_features
                NFS = len(selec_feats_thr)
                if NFS==0:
                    break
                X_train_val_bayes = X_train_val[:,selec_bool]

                # Optimize with BayesianOptimization
                tic = time.time()
                optimizer = BayesianOptimization(f=optimize_fun_bayes,
                                                pbounds=PBOUNDS,
                                                random_state=1234*run_total,
                                                allow_duplicate_points=True)
                optimizer.maximize(init_points=10, n_iter=110)
                tac = time.time()
                elapsed_time = (tac - tic) / 60.0
                print('Elapsed_time:', elapsed_time)
                print('Best:', optimizer.max)
                alpha=optimizer.max["params"]["alpha"]
                gamma=optimizer.max["params"]["gamma"]
                model = KernelRidge(alpha=10.0**alpha, gamma=10.0**gamma, kernel='rbf')
                model.fit(X_train_val_bayes, y_train_val)
                preds = model.predict(X_test[:,selec_bool])
                FINAL_RMSE_VAL = -optimize_fun_bayes(alpha=alpha, gamma=gamma)
                FINAL_RMSE_TST = mean_squared_error(y_test, preds, squared=False)
                print(f'THR={thr_features} NFS={NFS} FINAL_RMSE_VAL={FINAL_RMSE_VAL} FINAL_RMSE_TST={FINAL_RMSE_TST}')
                res_bayes.append(dict(name_db=name_db,
                                      run_total=run_total,
                                thr_features=thr_features,
                                NFS=NFS,
                                FINAL_RMSE_VAL=FINAL_RMSE_VAL,
                                FINAL_RMSE_TST=FINAL_RMSE_TST,
                                model = model,
                                selec_feats_thr=selec_feats_thr))
            res_bayes_df = pd.DataFrame(res_bayes)
            res_bayes_df.to_csv(DIR_SALIDA + 'bayes_thr_' + NAME_FILE + '.csv')
            print(res_bayes_df)



