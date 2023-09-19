# import openml
import numpy as np
import pandas as pd
import random, os, gc
from tqdm.notebook import tqdm
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, KFold
#from hybparsimony import HYBparsimony
import time
from datetime import datetime
from bayes_opt import BayesianOptimization


NUM_GEN = 250 #2
time_limit = 120.0 #0.10
NUM_RUNS = 5 #1
NUM_INI_ITERS =  10 #3


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# Optimize with BayesianOptimization
def optimize_fun_regression(alpha, gamma):
    global X_train_val, y_train_val, run_total
    model = KernelRidge(alpha=10.0**alpha, gamma=10.0**gamma, kernel='rbf')
    cv = KFold(n_splits=5, random_state=1234*run_total, shuffle=True)
    return np.mean(cross_val_score(model, X_train_val, y_train_val, cv=cv, scoring="neg_mean_squared_error"))

def optimize_fun_logloss(C):
    global X_train_val, y_train_val, run_total
    model = LogisticRegression(C=10.0**C, max_iter=5000)
    cv = KFold(n_splits=5, random_state=1234*run_total, shuffle=True)
    return np.mean(cross_val_score(model, X_train_val, y_train_val, cv=cv, scoring="neg_log_loss"))

def optimize_fun_f1macro(C):
    global X_train_val, y_train_val, run_total
    model = LogisticRegression(C=10.0**C, max_iter=5000)
    cv = KFold(n_splits=5, random_state=1234*run_total, shuffle=True)
    return np.mean(cross_val_score(model, X_train_val, y_train_val, cv=cv, scoring="f1_macro"))


now = datetime.now()
NAME_FILE = 'bayesian_opt_all_feats' + now.strftime("%Y_%m_%d_%H_%M_%S_") + '.csv'
    
res_basedata = pd.read_csv('./input/res_basedata.csv')
print(res_basedata.shape)
res_basedata = res_basedata.query('NFs<500').reset_index(drop=True)
print(res_basedata.shape)
print(res_basedata.type_prob.value_counts())
print(res_basedata.sort_values('NFs', ascending=False).head(10))

res_basedata = res_basedata.sort_values(['type_prob', 'num_classes'])

res_iters = []
for type_prob in ['binary', 'regression', 'multiclass']:
    for nrow, row in res_basedata.iterrows():
        if row['type_prob']==type_prob:
            train_val_size = row['len_y']//2
            test_size = row['len_y'] - train_val_size
            X = pd.read_csv('./datasets/'+row['name_file'])
            y = X['target_end']
            X.drop(columns='target_end', inplace=True)
            input_names = X.columns
            X_train_val, X_test, y_train_val, y_test = train_test_split(X.values, y.values,
                                                                        train_size = train_val_size, 
                                                                        random_state=12345)
            for run_total in range(NUM_RUNS):
                seed_everything(run_total*100)
                tic = time.time()
                print(row['name_ds'], row['type_prob'], row['nrows'], row['NFs'], row['num_classes'])
                if row['type_prob']=='regression':
                    algorithm='KernelRidge'
                    optimize_fun_bayes=optimize_fun_regression
                    PBOUNDS = dict(alpha=(-5, 5), gamma=(-5,3))
                elif row['type_prob']=='binary':
                    algorithm='LogisticRegression'
                    optimize_fun_bayes=optimize_fun_logloss
                    PBOUNDS = dict(C=(-5, 3))
                else:
                    algorithm='LogisticRegression'
                    optimize_fun_bayes=optimize_fun_f1macro
                    PBOUNDS = dict(C=(-5, 3))
                    
                try:
                    # Bayesian Optimization
                    # ---------------------
                    optimizer = BayesianOptimization(f=optimize_fun_bayes,
                                                pbounds=PBOUNDS,
                                                random_state=1234*run_total,
                                                allow_duplicate_points=True)
                    optimizer.maximize(init_points=NUM_INI_ITERS, n_iter=NUM_GEN)
      
                    NFS = row['NFs']
                    if row['type_prob']=='regression':
                        print('Best:', optimizer.max)
                        alpha=optimizer.max["params"]["alpha"]
                        gamma=optimizer.max["params"]["gamma"]
                        score_type = 'MSE/RMSE'
                        model = KernelRidge(alpha=10.0**alpha, gamma=10.0**gamma, kernel='rbf')
                        model.fit(X_train_val, y_train_val)
                        preds = model.predict(X_test)
                        FINAL_SCORE_VAL = -optimize_fun_bayes(alpha=alpha, gamma=gamma)
                        FINAL_SCORE_TST = round(mean_squared_error(y_test, preds, squared=False),6)
                    
                    if row['type_prob']=='binary':
                        print('Best:', optimizer.max)
                        C=optimizer.max["params"]["C"]
                        score_type = 'logloss'
                        model = LogisticRegression(C=10.0**C, max_iter=5000)
                        model.fit(X_train_val, y_train_val)
                        preds = model.predict_proba(X_test)[:,1]
                        FINAL_SCORE_VAL = -optimize_fun_bayes(C=C)
                        FINAL_SCORE_TST = round(log_loss(y_test, preds),6)
                        
                    if row['type_prob']=='multiclass':
                        print('Best:', optimizer.max)
                        C=optimizer.max["params"]["C"]
                        score_type = 'f1_macro'
                        model = LogisticRegression(C=10.0**C, max_iter=5000)
                        model.fit(X_train_val, y_train_val)
                        preds = model.predict(X_test)
                        FINAL_SCORE_VAL = optimize_fun_bayes(C=C)
                        FINAL_SCORE_TST = round(f1_score(y_test, preds, average="macro"),6)
                        
                    tac = time.time()
                    elapsed_time = (tac - tic) / 60.0
                    print('Elapsed_time:', elapsed_time)
                    print(f'DS={row["name_ds"]} run={run_total} type={row["type_prob"]} NFS={NFS} FINAL_SCORE_VAL={FINAL_SCORE_VAL} FINAL_SCORE_TST={FINAL_SCORE_TST}')
                    
                    res_iters.append(dict(
                                        optim='BO',
                                        name_db=row['name_ds'],
                                        type_prob=row['type_prob'],
                                        score_type=score_type,
                                        run_total=run_total,
                                        time_limit=time_limit,
                                        elapsed_time=elapsed_time,
                                        num_rows=row['nrows'],
                                        num_cols=row['NFs'],
                                        num_classes=row['num_classes'],
                                        train_val_size=train_val_size, 
                                        test_size=test_size,
                                        NFS=NFS,
                                        FINAL_SCORE_VAL=FINAL_SCORE_VAL,
                                        FINAL_SCORE_TST=FINAL_SCORE_TST,
                                        name_file=row['name_file'],
                                        model = model))

                    os.system('clear')
                    res_df = pd.DataFrame(res_iters)
                    res_df.to_csv('./results/'+NAME_FILE, index=False)
                    print("---------------------------------------------------------------------------")
                    print(res_df[['name_db','type_prob','score_type','num_rows','num_cols','num_classes',
                                'run_total','elapsed_time', 'NFS', 
                                'FINAL_SCORE_VAL', 'FINAL_SCORE_TST']])
                    print("---------------------------------------------------------------------------")
                    gc.collect()
                    time.sleep(3)
                except:
                    continue

            