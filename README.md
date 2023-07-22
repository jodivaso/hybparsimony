# HYBparsimony

[![PyPI version](https://badge.fury.io/py/GAparsimony.svg)](https://badge.fury.io/py/GAparsimony)
[![Documentation Status](https://readthedocs.org/projects/gaparsimony/badge/?version=latest)](https://gaparsimony.readthedocs.io/en/latest/?badge=latest)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=GAparsimony&metric=alert_status)](https://sonarcloud.io/dashboard?id=GAparsimony)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=GAparsimony&metric=bugs)](https://sonarcloud.io/dashboard?id=GAparsimony)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=GAparsimony&metric=code_smells)](https://sonarcloud.io/dashboard?id=GAparsimony)


HYBparsimony
===========

[Documentation](https://gaparsimony.readthedocs.io/en/latest/index.html)

**HYBparsimony** for Python is a package **for searching accurate parsimonious models by combining feature selection (FS), model
hyperparameter optimization (HO), and parsimonious model selection (PMS) based on a separate cost and complexity evaluation**.

To improve the search for parsimony, the hybrid method combines GA mechanisms such as selection, crossover and mutation within a PSO-based optimization algorithm that includes a strategy in which the best position of each particle (thus also the best position of each neighborhood) is calculated taking into account not only the goodness-of-fit, but also the parsimony principle. 

In HYBparsimony, the percentage of variables to be replaced with GA at each iteration $t$ is selected by a decreasing exponential function:
 $pcrossover=max(0.80 \cdot e^{(-\Gamma \cdot t)}, 0.10)$, that is adjusted by a $\Gamma$ parameter (by default $\Gamma$ is set to $0.50$). Thus, in the first iterations parsimony is promoted by GA mechanisms, i.e., replacing by crossover a high percentage of particles at the beginning. Subsequently, optimization with PSO becomes more relevant for the improvement of model accuracy. This differs from other hybrid methods in which the crossover is applied between the best individual position of each particle or other approaches in which the worst particles are also replaced by new particles, but at extreme positions.

Experiments show that, in general, and with a suitable $\Gamma$, HYBparsimony allows to obtain better, more parsimonious and more robust models compared to other methods. It also reduces the number of iterations and, consequently, the computational effort.

Installation
------------
Install these packages, [pip](https://pypi.org/project/HYBparsimony/):
``` {.bash}
pip install HYBparsimony
```
To install the current development version, you need to clone the repository and run :
``` {.bash}
python -m pip install << path to cloned repository >>
```

How to use this package
-----------------------

### Example 1: Regression

This example shows how to search with *HYBparsimony* package for a parsimonious (with low complexity) *KernelRidge* with *rbf* kernel model and for the *diabetes* dataset. *HYBparsimony* searches for the best input features and *KernelRidge* hyperparameters: $alpha$ and $gamma$. Models are evaluated by default with a 5-fold CV negative mean squared error (*Neg MSE*). Finally, root mean squared error (*$RMSE*) is calculated with another test dataset to check the degree of model generalization.

In this example, *rerank\_error* is set to $0.001$, but other values could improve the balance between model complexity and accuracy. PMS considers the most parsimonious model with the fewest number of features. The default complexity is $M_c = 10^9{N_{FS}} + int_{comp}$  where ${N_{FS}}$ is the number of selected input features and $int_{comp}$ is the internal measure of model complexity, which depends on the algorithm used for training. In this example, $int_{comp}$ for *KernelRidge* is measured by the sum of the squared coefficients. Therefore, between two models with the same number of features, the smaller sum of the squared weights will determine the more parsimonious model (smaller weights reduce the propagation of perturbations and improve robustness).


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from HYBparsimony import HYBparsimony

# Load 'diabetes' dataset
diabetes = load_diabetes()

X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

# Standarize X and y
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

# Search the best hyperparameters and features 
# (increasing 'time_limit' to improve RMSE with high consuming algorithms)
HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)
```
In each iteration, the first row shows the score and complexity of the best model. The second row shows the average score, and the score and complexity of the best model obtained in the same iteration. The values to the left of the first comma of the complexity correspond to the number of features (${N_{FS}}$).
```
Running iteration 0
Best model -> Score = -0.510786 Complexity = 9,017,405,352.5 
Iter = 0 -> MeanVal = -0.88274  ValBest = -0.510786   ComplexBest = 9,017,405,352.5 Time(min) = 0.005858

Running iteration 1
Best model -> Score = -0.499005 Complexity = 8,000,032,783.88 
Iter = 1 -> MeanVal = -0.659969  ValBest = -0.499005   ComplexBest = 8,000,032,783.88 Time(min) = 0.004452

...
...
...

Running iteration 34
Best model -> Score = -0.489468 Complexity = 8,000,002,255.68 
Iter = 34 -> MeanVal = -0.527314  ValBest = -0.489468   ComplexBest = 8,000,002,255.68 Time(min) = 0.007533

Running iteration 35
Best model -> Score = -0.489457 Complexity = 8,000,002,199.12 
Iter = 35 -> MeanVal = -0.526294  ValBest = -0.489457   ComplexBest = 8,000,002,199.12 Time(min) = 0.006522

Time limit reached. Stopped.
```

Show final results:

```python
preds = HYBparsimony_model.predict(X_test)
print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
print(f'Selected features:{HYBparsimony_model.selected_features}')
print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
print(f'5-CV MSE = {-round(HYBparsimony_model.best_score,6)}')
print(f'RMSE test = {round(mean_squared_error(y_test, preds, squared=False),6)}')
```

```
Best Model = KernelRidge(alpha=0.26747155972470016, gamma=0.010478997542788611, kernel='rbf')
Selected features:['age' 'sex' 'bmi' 'bp' 's1' 's4' 's5' 's6']
Complexity = 8,000,002,199.12
5-CV MSE = 0.489457
RMSE test = 0.681918
```

Compare with different algorithms:

```python
algorithms_reg = ['Ridge', 'Lasso', 'KernelRidge', 'KNeighborsRegressor', 'MLPRegressor', 'SVR',
'DecisionTreeRegressor', 'RandomForestRegressor']
res = []
for algo in algorithms_reg:
    print('#######################')
    print('Searching best: ', algo)
    HYBparsimony_model = HYBparsimony(algorithm=algo,
                                      features=diabetes.feature_names,
                                      rerank_error=0.001,
                                      verbose=1)
    # Search the best hyperparameters and features 
    # (increasing 'time_limit' to improve RMSE with high consuming algorithms)
    HYBparsimony_model.fit(X_train, y_train, time_limit=5)
    # Check results with test dataset
    preds = HYBparsimony_model.predict(X_test)
    print(algo, "RMSE test", mean_squared_error(y_test, preds, squared=False))
    print('Selected features:',HYBparsimony_model.selected_features)
    print(HYBparsimony_model.best_model)
    print('#######################')
    # Append results
    res.append(dict(algo=algo,
                    MSE_5CV= -round(HYBparsimony_model.best_score,6),
                    RMSE=round(mean_squared_error(y_test, preds, squared=False),6),
                    NFS=HYBparsimony_model.best_complexity//1e9,
                    selected_features = HYBparsimony_model.selected_features,
                    best_model=HYBparsimony_model.best_model))
res = pd.DataFrame(res).sort_values('RMSE')
res.to_csv('res_models.csv')
# Visualize results
print(res[['best_model', 'MSE_5CV', 'RMSE', 'NFS', 'selected_features']])
```

We obtain the following results:

```
                    algo   MSE_5CV      RMSE  NFS
4           MLPRegressor  0.491437  0.673157    7
2            KernelRidge  0.488908  0.679108    7
1                  Lasso  0.495795  0.694631    8
0                  Ridge  0.495662  0.694885    8
5                    SVR  0.487899  0.696137    7
3    KNeighborsRegressor  0.523190  0.705371    6
7  RandomForestRegressor  0.546012  0.761268    8
6  DecisionTreeRegressor  0.630503  0.864194    3
```
However, if we increase the time limit to 60 minutes, the maximum number of iterations and use a more robust validation with a 10-repeated 5-fold crossvalidation.

```python
 HYBparsimony_model = HYBparsimony(algorithm=algo,
                                   features=diabetes.feature_names,
                                   rerank_error=0.001,
                                   cv=RepeatedKFold(n_repeats=10, n_splits=5),
                                   maxiter=1000,
                                   verbose=1)
HYBparsimony_model.fit(X_train, y_train, time_limit=60)
```
We can improve results in RMSE and parsimony.

|Algorithm|MSE\_10R5CV|RMSEtst|NFS|selected\_features|best\_model|
|-|-|-|-|-|-|
|**MLPRegressor**|0.493201|**0.671856**|**6**|['sex' 'bmi' 'bp' 's1' 's2' 's5']|MLPRegressor(activation='logistic', alpha=0.010729877296924203, hidden_layer_sizes=1, max_iter=5000, n_iter_no_change=20, random_state=1234, solver='lbfgs', tol=1e-05)|\
|KernelRidge|0.483465|0.679036|7|['age' 'sex' 'bmi' 'bp' 's3' 's5' 's6']|KernelRidge(alpha=0.3664462701238023, gamma=0.01808883688516421, kernel='rbf')|\
|SVR|0.487392|0.682699|8|['age' 'sex' 'bmi' 'bp' 's1' 's4' 's5' 's6']|SVR(C=0.8476135773996406, gamma=0.02324169209860404)|\
|KNeighborsRegressor|0.521326|0.687740|6|['sex' 'bmi' 'bp' 's3' 's5' 's6']|KNeighborsRegressor(n\_neighbors=11)|\
|Lasso|0.493825|0.696194|7|['sex' 'bmi' 'bp' 's1' 's2' 's5' 's6']|Lasso(alpha=0.0002735058905983914)|\
|Ridge|0.492570|0.696273|7|['sex' 'bmi' 'bp' 's1' 's2' 's5' 's6']|Ridge(alpha=0.1321381563140431)|\
|RandomForestRegressor|0.552005|0.703769|9|['age' 'sex' 'bmi' 'bp' 's2' 's3' 's4' 's5' 's6']|RandomForestRegressor(max_depth=17, min_samples_split=25, n_estimators=473)|\
|DecisionTreeRegressor|0.628316|0.864194|5|['age' 'sex' 'bmi' 's4' 's6']|DecisionTreeRegressor(max_depth=2, min_samples_split=20)|\


### Example 2: Binary Classification

This example shows how to use *HYBparsimony* in a binary classification problem with *breast_cancer* dataset. By default, *LogisticRegression* algorithm and *neg_log_loss* scoring is selected.

```python
 import pandas as pd
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
 from sklearn.datasets import load_breast_cancer
 from sklearn.metrics import log_loss
 from HYBparsimony import HYBparsimony
 
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
                                   rerank_error=0.005,
                                   verbose=1)
 HYBparsimony_model.fit(X_train, y_train, time_limit=0.50)
 # Extract probs of class==1
 preds = HYBparsimony_model.predict_proba(X_test)[:,1]
 print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
 print(f'Selected features:{HYBparsimony_model.selected_features}')
 print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
 print(f'5-CV logloss = {-round(HYBparsimony_model.best_score,6)}')
 print(f'logloss test = {round(log_loss(y_test, preds),6)}')
```

In this example, best model is obtained with 11 features from the 30 original inputs. 

```
(569, 30)
Detected a binary-class problem. Using 'neg_log_loss' as default scoring function.
Running iteration 0
Best model -> Score = -0.091519 Complexity = 29,000,000,005.11 
Iter = 0 -> MeanVal = -0.297448  ValBest = -0.091519   ComplexBest = 29,000,000,005.11 Time(min) = 0.006501

Running iteration 1
Best model -> Score = -0.085673 Complexity = 27,000,000,009.97 
Iter = 1 -> MeanVal = -0.117216  ValBest = -0.085673   ComplexBest = 27,000,000,009.97 Time(min) = 0.004273

...
...

Running iteration 102
Best model -> Score = -0.064557 Complexity = 11,000,000,039.47 
Iter = 102 -> MeanVal = -0.076314  ValBest = -0.066261   ComplexBest = 9,000,000,047.25 Time(min) = 0.004769

Running iteration 103
Best model -> Score = -0.064557 Complexity = 11,000,000,039.47 
Iter = 103 -> MeanVal = -0.086243  ValBest = -0.064995   ComplexBest = 11,000,000,031.2 Time(min) = 0.004591

Time limit reached. Stopped.

Best Model = LogisticRegression(C=5.92705799354935)
Selected features:['mean texture' 'mean concave points' 'radius error' 'area error'
 'compactness error' 'worst radius' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst concavity' 'worst symmetry']
Complexity = 11,000,000,039.47
5-CV logloss = 0.064557
logloss test = 0.076254
```

However, with small databases like *breast_cancer*, it is highly recommended to use a repeated cross-validation and execute
*HYBparsimony** with different seeds in order to find the most important input features and best model hyper-parameters.

We also can compare with other algorithms.

```python

algorithms_clas = ['LogisticRegression', 'MLPClassifier', 
                    'SVC', 'DecisionTreeClassifier',
                    'RandomForestClassifier', 'KNeighborsClassifier',
                    ]
    res = []
    for algo in algorithms_clas:
        print('#######################')
        print('Searching best: ', algo)
        HYBparsimony_model = HYBparsimony(algorithm=algo,
                                          features=breast_cancer.feature_names,
                                          rerank_error=0.005,
                                          cv=RepeatedKFold(n_splits=5, n_repeats=10),
                                          maxiter=1000,
                                          verbose=1)
        # Search the best hyperparameters and features 
        # (increasing 'time_limit' to improve neg_log_loss with high consuming algorithms)
        HYBparsimony_model.fit(X_train, y_train, time_limit=60.0)
        # Check results with test dataset
        preds = HYBparsimony_model.predict_proba(X_test)[:,1]
        print(algo, "Logloss_Test=", round(log_loss(y_test, preds),6))
        print('Selected features:',HYBparsimony_model.selected_features)
        print(HYBparsimony_model.best_model)
        print('#######################')
        # Append results
        res.append(dict(algo=algo,
                        Logloss_10R5CV= -round(HYBparsimony_model.best_score,6),
                        Logloss_Test = round(log_loss(y_test, preds),6),
                        NFS=int(HYBparsimony_model.best_complexity//1e9),
                        selected_features = HYBparsimony_model.selected_features,
                        best_model=HYBparsimony_model.best_model))
    res = pd.DataFrame(res).sort_values('Logloss_Test')
    res.to_csv('res_models_class.csv')
    # Visualize results
    print(res[['algo', 'Logloss_10R5CV', 'Logloss_Test', 'NFS']])
```

|algo|Logloss\_10R5CV|Logloss\_Test|NFS|selected\_features|best\_model|
|-|-|-|-|-|-|
|LogisticRegression|0.066868|0.079512|10|'radius error','smoothness error','compactness error','worst radius','worst texture','worst perimeter','worst area','worst concavity','worst concave points','worst symmetry'|LogisticRegression(C=2.5457613022710692)|
|SVC|0.061924|0.093283|9|'mean texture','radius error','smoothness error','compactness error','symmetry error','worst perimeter','worst concavity','worst concave points','worst fractal dimension'|SVC(C=10.017400170851333, gamma=0.030271440833644657, probability=True)|
|MLPClassifier|0.055662|0.100951|14|'mean smoothness','mean compactness','mean concavity','texture error','area error','smoothness error','concave points error','fractal dimension error','worst radius','worst texture','worst perimeter','worst area','worst compactness','worst fractal dimension'|MLPClassifier(activation='logistic', alpha=0.08468913411920591, hidden\_layer\_sizes=8, max\_iter=5000, n\_iter\_no\_change=20, random\_state=1234, solver='lbfgs', tol=1e-05)|
|DecisionTreeClassifier|0.214163|0.304484|7|'mean radius','mean compactness','mean concave points','worst texture','worst smoothness','worst symmetry','worst fractal dimension'|DecisionTreeClassifier(max\_depth=2, min\_samples\_split=19)|
|RandomForestClassifier|0.098342|0.4229|12|'mean texture','mean smoothness','mean concave points','area error','smoothness error','compactness error','worst texture','worst area','worst smoothness','worst concave points','worst symmetry','worst fractal dimension'|RandomForestClassifier(max\_depth=20, n\_estimators=126)|
|KNeighborsClassifier|0.079658|0.714111|17|'mean radius','mean texture','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','radius error','perimeter error','area error','smoothness error','compactness error','symmetry error','worst radius','worst texture','worst compactness','worst concave points'|KNeighborsClassifier(n\_neighbors=7, p=1)|



References
----------
F.J. Martinez-de-Pison, J. Ferreiro, E. Fraile, A. Pernia-Espinoza, A comparative study of six model complexity 
metrics to search for parsimonious models with GAparsimony R Package, Neurocomputing,
Volume 452, 2021, Pages 317-332, ISSN 0925-2312, [https://doi.org/10.1016/j.neucom.2020.02.135](https://doi.org/10.1016/j.neucom.2020.02.135).

Martinez-de-Pison, F.J., Gonzalez-Sendino, R., Aldama, A., Ferreiro-Cabello, J., Fraile-Garcia, E. Hybrid methodology 
based on Bayesian optimization and GA-PARSIMONY to search for parsimony models by combining hyperparameter optimization 
and feature selection (2019) Neurocomputing, 354, pp. 20-26. [https://doi.org/10.1016/j.neucom.2018.05.136](https://doi.org/10.1016/j.neucom.2018.05.136).

Urraca R., Sodupe-Ortega E., Antonanzas E., Antonanzas-Torres F., Martinez-de-Pison, F.J. (2017). Evaluation of a 
novel GA-based methodology for model structure selection: The GA-PARSIMONY. Neurocomputing, Online July 2017. [https://doi.org/10.1016/j.neucom.2016.08.154](https://doi.org/10.1016/j.neucom.2016.08.154).

Martinez-De-Pison, F.J., Gonzalez-Sendino, R., Ferreiro, J., Fraile, E., Pernia-Espinoza, A. GAparsimony: An R 
package for searching parsimonious models by combining hyperparameter optimization and feature selection (2018) Lecture 
Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 
10870 LNAI, pp. 62-73. [https://doi.org/10.1007/978-3-319-92639-1_6](https://doi.org/10.1007/978-3-319-92639-1_6).
