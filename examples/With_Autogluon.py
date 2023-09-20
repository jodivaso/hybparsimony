import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss
from hybparsimony import HYBparsimony
from autogluon.tabular import TabularDataset, TabularPredictor
from hybparsimony import util

if __name__ == "__main__":

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
    #                                 rerank_error=0.005,
    #                                 verbose=1)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.50)
    # # Extract probs of class==1
    # preds = HYBparsimony_model.predict_proba(X_test)[:,1]
    # print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
    # print(f'Selected features:{HYBparsimony_model.selected_features}')
    # print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
    # print(f'5-CV logloss = {-round(HYBparsimony_model.best_score,6)}')
    # print(f'logloss test = {round(log_loss(y_test, preds),6)}')
    
    # # Use autogluon using the 30 input features
    # X_train_df = pd.DataFrame(np.hstack([X_train, y_train.reshape(-1,1).astype(int)]))
    # X_test_df = pd.DataFrame(np.hstack([X_test, y_test.reshape(-1,1).astype(int)]))
    # print(X_train_df.shape, X_test_df.shape)
    # X_train_df.columns = list(breast_cancer.feature_names)+['target']
    # X_test_df.columns = list(breast_cancer.feature_names)+['target']
    # print(X_train_df.head())
    
    # predictor = TabularPredictor(label='target').fit(X_train_df)
    # y_pred = predictor.predict(X_test_df.drop(columns=['target']))
    # print(f'Logloss test with AUTOGLUON (all feats) = {round(log_loss(y_test, y_pred),6)}')
    
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    label = 'class'
    print(train_data.shape)
    
    
    # Train with features
    subsample_size = 2000  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = train_data.sample(n=subsample_size, random_state=0)
    train_data[label] = train_data[label].map({' >50K':1, ' <=50K':0}).astype(int)
    train_data_nolab = train_data.drop(columns=[label])
    save_path = 'agModels-predictClass'
    # predictor = TabularPredictor(label=label, path=save_path, eval_metric='log_loss').fit(train_data, time_limit=30)
    
    # Shows performance with a new dataset
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    test_data[label] = test_data[label].map({' >50K':1, ' <=50K':0}).astype(int)
    print(test_data.shape)
    y_test = test_data[label]  # values to predict
    test_data_nolab = test_data.drop(columns=[label])
    # y_pred = predictor.predict_proba(test_data_nolab)
    # Log_Loss = log_loss(y_true=y_test, y_pred=y_pred)
    # print(f'Log_loss with test usign all features={Log_Loss}')
    # print('##################################################')
    
    
    
    def custom_fun_logloss(estimator, X, y):
        x_train_custom, x_test_custom, y_train_custom, y_test_custom = train_test_split(X, y, 
                                                                                        test_size=0.20, 
                                                                                        shuffle=True, 
                                                                                        random_state=0)
        return np.array([-1.0])
        # X_train_df = pd.DataFrame(np.hstack([x_train_custom, y_train_custom.reshape(-1,1).astype(int)]))
        # X_test_df = pd.DataFrame(x_test_custom)
        # print('ENTRO')
        # predictor = estimator(label=label, path=save_path, eval_metric='log_loss').fit(X_train_df, time_limit=30)
        # y_pred = predictor.predict_proba(X_test_df)
        # return np.array([-log_loss(y_true=y_test_custom, y_pred=y_pred)])
    
    Autogluon_for_HYB = {"estimator": TabularPredictor,
                         "complexity": util.complexity.generic_complexity,
                         "C": {"range": (-5, 3), "type": util.Population.POWER}
    }
     
    input_names = train_data_nolab.columns
    HYBparsimony_model = HYBparsimony(
                                    algorithm=Autogluon_for_HYB,
                                    scoring='neg_log_loss',
                                    custom_eval_fun=custom_fun_logloss,
                                    features=input_names,
                                    rerank_error=0.001,
                                    gamma_crossover=0.50,
                                    seed_ini=0,
                                    npart=15,
                                    maxiter=100,
                                    early_stop=35,
                                    verbose=1,
                                    n_jobs=1)
    HYBparsimony_model.fit(train_data_nolab, train_data[label])   
    best_model_parameters = HYBparsimony_model.best_model_conf[:-len(input_names)]
    best_model_probsfeats = HYBparsimony_model.best_model_conf[-len(input_names):]
    print(best_model_parameters)
    
    
