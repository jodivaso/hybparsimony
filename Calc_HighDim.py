import copy
import os
import pandas as pd
import numpy as np
import math
import glob, random, time, sys
from string import printable
from datetime import datetime

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, KFold
from sklearn.metrics import make_scorer
from sklearn.datasets import load_diabetes, load_iris

from HYBparsimony import Population, HYBparsimony, order
from HYBparsimony.hybparsimony import default_cv_score_classification
from HYBparsimony.util import knn_complexity

# from PSOparsimony_Nested import PSOparsimony
from bayes_opt import BayesianOptimization
# from HYBparsimony import HYBparsimony
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import __version__ as sk_version

DIR_SALIDA = 'experiments_23abr2023/'
NUM_RUNS = 5

tiempos = [['slice_norm_reduc.csv', 3, 2000, 378, 0.50], 
           ['blog_norm.csv', 3, 2000, 276, 0.50], 
        #    ['crime_norm.csv', 3, 2000, 127, 0.50], 
        #    ['tecator_norm.csv', 3, 2000, 124, 0.50],
        #    ['ailerons_norm.csv', 3, 2000, 40, 0.50], 
        #    ['puma_norm.csv', 3, 2000, 32, 0.50],
        #    ['bank_norm.csv', 3, 2000, 32, 0.50], 
        #    ['pol_norm.csv', 3, 2000, 26, 0.50],
        #    ['cpu_act_norm.csv', 3, 2000, 21, 0.50],
        #    ['elevators_norm.csv', 3, 2000, 18, 0.50],
        #    ['meta_norm.csv', 3, 2000, 17, 0.50],
        #    ['bodyfat_norm.csv', 3, 2000, 14, 0.50], 
        #    ['boston_norm.csv', 3, 2000, 13, 0.50],
        #    ['housing_norm.csv', 3, 2000, 13, 0.50],
        #    ['concrete_norm.csv', 3, 2000, 8, 0.50],
        #    ['no2_norm.csv', 3, 2000, 7, 0.50],
        #    ['pm10_norm.csv', 3, 2000, 7, 0.50],
        #    ['strike_norm.csv', 3, 2000, 6, 0.50]
          ]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def rmse_func(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))



if __name__ == "__main__":
    print(sk_version, pd.__version__, np.__version__)
    now = datetime.now()
    NAME_FILE = 'hyb_all_dbs' + now.strftime("%Y_%m_%d_%H_%M_%S_") 


    if not os.path.exists(DIR_SALIDA):
        os.mkdir(DIR_SALIDA)
    
    # Datasets loop
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
        print(f'Hyb gamma_crossover={gamma_crossover} name_db={name_db} time_limit={time_limit} num_cols={num_cols} train_val_size={train_val_size} test_size={test_size}')
        
   
        input_names, target_name = df_VAL.columns[:-1], df_VAL.columns[-1]
        X, y = df_VAL[input_names], df_VAL[target_name]

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, 
                                                                    train_size = names_tiempo[2] / len(X), 
                                                                    random_state=12345)
        scaler_x = StandardScaler()
        X_train_val = pd.DataFrame(scaler_x.fit_transform(X_train_val), columns=input_names)
        X_test = pd.DataFrame(scaler_x.transform(X_test), columns=input_names)
        
        scaler_y = StandardScaler()
        y_train_val = pd.DataFrame(scaler_y.fit_transform(y_train_val.values.reshape(-1, 1)), columns=[target_name])
        y_test = pd.DataFrame(scaler_y.transform(y_test.values.reshape(-1, 1)), columns=[target_name])
   
        # Main Loop
        out_res = []
        probs_feats = []
        for num_run in range(NUM_RUNS):
            seed_everything(num_run)
            HYBparsimony_model = HYBparsimony(algorithm='Ridge',
                                            features=input_names,
                                            rerank_error=0.001,
                                            gamma_crossover=gamma_crossover,
                                            seed_ini=num_run,
                                            verbose=0)
            HYBparsimony_model.fit(X_train_val.values, y_train_val.values, time_limit=0.20)
            preds = HYBparsimony_model.predict(X_test.values)
            RMSE_test = mean_squared_error(y_test, preds, squared=False)
            print(f'{NAME_FILE} name_db={name_db} run={num_run} RMSE_test={RMSE_test}')
            print(HYBparsimony_model.best_model)
            best_model_parameters = HYBparsimony_model.best_model_conf[:-len(input_names)]
            best_model_probsfeats = HYBparsimony_model.best_model_conf[-len(input_names):]
            out_res.append(dict(name_db=name_db, num_run=num_run,
                                time_limit=time_limit, 
                                num_cols=num_cols, train_val_size=train_val_size, test_size=test_size,
                                RMSE=RMSE_test,
                                best_model_parameters=best_model_parameters,
                                best_model_probsfeats=best_model_probsfeats))
            probs_feats.append(best_model_probsfeats)
        probs_feats = pd.DataFrame(probs_feats, columns=input_names)
        probs_feats.to_csv(DIR_SALIDA + 'probs_' + name_db + '__' + NAME_FILE + '.csv', index=False)
            
    out_res = pd.DataFrame(out_res)
    print(out_res.head())
    out_res.to_csv(DIR_SALIDA + NAME_FILE + '.csv', index=False)






