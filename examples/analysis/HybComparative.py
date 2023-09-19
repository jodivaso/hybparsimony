import openml
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
from hybparsimony import HYBparsimony
from hybparsimony import util
import time
from datetime import datetime



NUM_GEN = 250 #2
time_limit = 120.0 #0.10
NUM_RUNS = 5 #1

LogisticRegression_new = {"estimator": LogisticRegression,
                    "complexity": util.complexity.linearModels_complexity,
                    "C": {"range": (-5, 3), "type": util.Population.POWER},
                    "max_iter": {"value": 5000, "type": util.Population.CONSTANT}
                    }

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

now = datetime.now()
NAME_FILE = 'hybrid_alone' + now.strftime("%Y_%m_%d_%H_%M_%S_") + '.csv'
    
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
                else:
                    algorithm=LogisticRegression_new
                try:
                    # HYBparsimony Model
                    # ------------------
                    HYBparsimony_model = HYBparsimony(algorithm=algorithm,
                                                        features=input_names,
                                                        rerank_error=0.001,
                                                        gamma_crossover=0.50,
                                                        seed_ini=1234*run_total,
                                                        npart=15,
                                                        maxiter=NUM_GEN,
                                                        early_stop=35,
                                                        verbose=1,
                                                        n_jobs=1)
                    HYBparsimony_model.fit(X_train_val, y_train_val, time_limit=time_limit)      
                    best_model_parameters = HYBparsimony_model.best_model_conf[:-len(input_names)]
                    best_model_probsfeats = HYBparsimony_model.best_model_conf[-len(input_names):]
                    selec_feats_thr = list(input_names[best_model_probsfeats>=0.50])
                    assert all(HYBparsimony_model.selected_features==selec_feats_thr)
                    selec_bool = best_model_probsfeats>=0.50
                    NFS = len(selec_feats_thr)
                    print(best_model_parameters[0])

                    if row['type_prob']=='regression':
                        score_type = 'MSE/RMSE'
                        model = KernelRidge(alpha=10.0**best_model_parameters[0], 
                                            gamma=10.0**best_model_parameters[1], 
                                            kernel='rbf')
                        model.fit(X_train_val[:,selec_bool], y_train_val)
                        preds = model.predict(X_test[:,selec_bool])
                        FINAL_SCORE_VAL = -round(HYBparsimony_model.best_score,6)
                        FINAL_SCORE_TST = round(mean_squared_error(y_test, preds, squared=False),6)
                    
                    if row['type_prob']=='binary':
                        score_type = 'logloss'
                        model = LogisticRegression(C=10.0**best_model_parameters[0],
                                                   max_iter=5000)
                        model.fit(X_train_val[:,selec_bool], y_train_val)
                        preds = model.predict_proba(X_test[:,selec_bool])[:,1]
                        FINAL_SCORE_VAL = -round(HYBparsimony_model.best_score,6)
                        FINAL_SCORE_TST = round(log_loss(y_test, preds),6)
                        
                    if row['type_prob']=='multiclass':
                        score_type = 'f1_macro'
                        model = LogisticRegression(C=10.0**best_model_parameters[0],
                                                   max_iter=5000)
                        model.fit(X_train_val[:,selec_bool], y_train_val)
                        preds = model.predict(X_test[:,selec_bool])
                        FINAL_SCORE_VAL = round(HYBparsimony_model.best_score,6)
                        FINAL_SCORE_TST = round(f1_score(y_test, preds, average="macro"),6)
                        
                    tac = time.time()
                    elapsed_time = (tac - tic) / 60.0
                    print('Elapsed_time:', elapsed_time)
                    print(f'DS={row["name_ds"]} run={run_total} type={row["type_prob"]} NFS={NFS} FINAL_SCORE_VAL={FINAL_SCORE_VAL} FINAL_SCORE_TST={FINAL_SCORE_TST}')
                    print(HYBparsimony_model.best_model)
                    
                    res_iters.append(dict(
                                        optim='HYB',
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
                                        model = model,
                                        selec_feats_thr=selec_feats_thr,
                                        best_model_parameters=best_model_parameters,
                                        best_model_probsfeats=best_model_probsfeats))
                    
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

            