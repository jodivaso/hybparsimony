# HYBparsimony
[![Python](https://img.shields.io/pypi/pyversions/hybparsimony)](https://img.shields.io/pypi/pyversions/hybparsimony)
[![PyPI version](https://img.shields.io/pypi/v/hybparsimony.svg)](https://img.shields.io/pypi/v/hybparsimony.svg)
[![Downloads](https://img.shields.io/pypi/dm/hybparsimony)](https://img.shields.io/pypi/dm/hybparsimony)
[![Documentation Status](https://readthedocs.org/projects/hybparsimony/badge/?version=latest)](https://hybparsimony.readthedocs.io/en/latest/?badge=latest)

[Documentation](https://hybparsimony.readthedocs.io/en/latest/index.html)

***HYBparsimony*** is a Python package that simultaneously performs: 
- **automatic feature selection** (FS)
- **automatic model hyperparameter optimization** (HO)
- **automatic parsimonious model selection** (PMS)

[Experiments with 100 datasets](https://github.com/jodivaso/hybparsimony/tree/master/examples/analysis) showed that ***HYBparsimony*** allows to obtain better, more parsimonious and more robust models compared to other methods, also reducing the number of iterations and the computational effort. [See the paper](https://doi.org/10.1016/j.neucom.2023.126840) for further details.

Its use is very simple:

```python
# Basic example with the Iris dataset (classification)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from hybparsimony import HYBparsimony

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

HYBparsimony_model = HYBparsimony()
HYBparsimony_model.fit(X_train, y_train, time_limit=0.1) # Train with time limit 0.1 minutes (6 seconds)
print(HYBparsimony_model.selected_features) # Print the selected features
print(HYBparsimony_model.best_model) # Print the model and its hyperparameters
print(HYBparsimony_model.best_score) # Print the score
```

In this example, the output model uses 3 of the 4 input variables, optimizing a *LogisticRegression* model.


[![Presentation of HYBparsimony (in spanish)](https://github.com/jodivaso/hybparsimony/blob/master/docs/Charla%233.png)](https://www.youtube.com/watch?v=F6R_ibMPQgg&t=0s "Presentation of HYBparsimony (in spanish)")

Installation
------------
Install the package using [pip](https://pypi.org/project/hybparsimony/):
``` {.bash}
pip install hybparsimony
```

Detailed explanation
------------

***HYBparsimony*** for Python is a package **for searching accurate parsimonious models by combining feature selection (FS), model hyperparameter optimization (HO), and parsimonious model selection (PMS) based on a separate cost and complexity evaluation** ([slices-HAIS2022](./docs/presentacion_HAIS2022_javi.pdf), [slices-HAIS2023](./docs/presentacion_HAIS2023_javi.pdf))

To improve the search for parsimony, the hybrid method combines GA mechanisms such as selection, crossover and mutation within a PSO-based optimization algorithm that includes a strategy in which the best position of each particle (thus also the best position of each neighborhood) is calculated taking into account not only the goodness-of-fit, but also the parsimony principle.

In HYBparsimony, the percentage of variables to be replaced with GA at each iteration $t$ is selected by a decreasing exponential function:
 $pcrossover=max(0.80 \cdot e^{(-\Gamma \cdot t)}, 0.10)$, that is adjusted by a $\Gamma$ parameter (by default $\Gamma$ is set to $0.50$). Thus, in the first iterations parsimony is promoted by GA mechanisms, i.e., replacing by crossover a high percentage of particles at the beginning. Subsequently, optimization with PSO becomes more relevant for the improvement of model accuracy. This differs from other hybrid methods in which the crossover is applied between the best individual position of each particle or other approaches in which the worst particles are also replaced by new particles, but at extreme positions.

[Experiments with 100 datasets](https://github.com/jodivaso/hybparsimony/tree/master/examples/analysis) showed that, in general, and with a suitable $\Gamma$, HYBparsimony allows to obtain better, more parsimonious and more robust models compared to other methods. It also reduces the number of iterations vs previous methods and, consequently, the computational effort.


Further examples: how to use this package
-----------------------
**Note 1**: The datasets used in these examples are of small size. These datasets have been selected in order to speed up the calculation process of the examples. With such small datasets it is necessary to use robust validation methods such as bootstrapping or repeated cross validation. It is also recommended to repeat the use of HYBparsimony with different random seeds in order to obtain more solid conclusions.

**Note 2**: The results of the examples may vary depending on the hardware available.


### Example 1: Regression

This example shows how to search with *hybparsimony* package for a parsimonious (with low complexity) *KernelRidge* with *rbf* kernel model and for the *diabetes* dataset. *HYBparsimony* searches for the best input features and *KernelRidge* hyperparameters: $alpha$ and $gamma$. Models are evaluated by default with a 5-fold CV negative mean squared error (*Neg MSE*). Finally, root mean squared error (*RMSE*) is calculated with another test dataset to check the degree of model generalization.

In this example, *rerank\_error* is set to $0.001$, but other values could improve the balance between model complexity and accuracy. PMS considers the most parsimonious model with the fewest number of features. The default complexity is $M_c = 10^9{N_{FS}} + int_{comp}$  where ${N_{FS}}$ is the number of selected input features and $int_{comp}$ is the internal measure of model complexity, which depends on the algorithm used for training. In this example, $int_{comp}$ for *KernelRidge* is measured by the sum of the squared coefficients. Therefore, between two models with the same number of features, the smaller sum of the squared weights will determine the more parsimonious model (smaller weights reduce the propagation of perturbations and improve robustness).

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from hybparsimony import HYBparsimony

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
                                n_jobs=1,
                                verbose=1)

# Search the best hyperparameters and features 
# (increasing 'time_limit' to improve RMSE with high consuming algorithms)
HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)
```
In each iteration, the first row shows the score and complexity of the best model. The second row shows the average score, and the score and complexity of the best model obtained in the same iteration. The values to the left of the first comma of the complexity correspond to the number of features (${N_{FS}}$), the rest is the internal complexity of the model.
```
Detected a regression problem. Using 'neg_mean_squared_error' as default scoring function.
Running iteration 0
Best model -> Score = -0.510786 Complexity = 9,017,405,352.5 
Iter = 0 -> MeanVal = -0.88274  ValBest = -0.510786   ComplexBest = 9,017,405,352.5 Time(min) = 0.014713

Running iteration 1
Best model -> Score = -0.499005 Complexity = 8,000,032,783.88 
Iter = 1 -> MeanVal = -0.659969  ValBest = -0.499005   ComplexBest = 8,000,032,783.88 Time(min) = 0.007782

...
...
...

Running iteration 45
Best model -> Score = -0.489457 Complexity = 8,000,002,199.12 
Iter = 45 -> MeanVal = -0.531697  ValBest = -0.490061   ComplexBest = 8,000,000,443.89 Time(min) = 0.003815

Running iteration 46
Best model -> Score = -0.489457 Complexity = 8,000,002,199.12 
Iter = 46 -> MeanVal = -0.541818  ValBest = -0.493126   ComplexBest = 7,000,030,081.16 Time(min) = 0.003704

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
# Visualize results
print(res[['best_model', 'MSE_5CV', 'RMSE', 'NFS', 'selected_features']])
```

We obtain the following results:


```
best_model      MSE_5CV   RMSE      NFS selected_features
4  MLPRegressor 0.491424  0.672799  7.0 [sex, bmi, bp, s1, s4, s5, s6]
2  KernelRidge  0.488908  0.679108  7.0 [age, sex, bmi, bp, s3, s5, s6]
1  Lasso        0.495795  0.694631  8.0 [sex, bmi, bp, s1, s2, s4, s5, s6]
0  Ridge        0.495662  0.694885  8.0 [sex, bmi, bp, s1, s2, s4, s5, s6]
5  SVR          0.487899  0.696137  7.0 [sex, bmi, bp, s1, s4, s5, s6]
3  KNeighbors   0.523190  0.705371  6.0 [sex, bmi, bp, s3, s5, s6]
7  RandomForest 0.535958  0.760138  6.0 [sex, bmi, s1, s4, s5, s6]
6  DecisionTree 0.625424  0.847182  3.0 [bmi, s4, s6]
```
However, we can improve results in RMSE and parsimony if we increase the time limit to 60 minutes, the maximum number of iterations to 1000, and use a more robust validation with a 10-repeated 5-fold crossvalidation.

```python
 HYBparsimony_model = HYBparsimony(algorithm=algo,
                                   features=diabetes.feature_names,
                                   rerank_error=0.001,
                                   cv=RepeatedKFold(n_repeats=10, n_splits=5),
                                   n_jobs=10, # each job executes one fold
                                   maxiter=1000,
                                   verbose=1)
HYBparsimony_model.fit(X_train, y_train, time_limit=60)
```
Note: *n_jobs* represents the number of CPU cores used within the *cross_val_score()* function included in *default_cv_score()*. Also, it is important to mention that certain *scikit-learn* algorithms inherently employ parallelization as well. Thus, with some algorithms it will be necessary to consider the sharing of cores between the algorithm and the cross_val_score() function.

The following table shows the best models found for each algorithm. In this case, **the model that best generalizes the problem is an ML regressor with only 6 features out of 10 and a single neuron in the hidden layer!**

|Algorithm|MSE\_10R5CV|RMSEtst|NFS|selected\_features|best\_model|
|-|-|-|-|-|-|
|**MLPRegressor**|0.493201|**0.671856**|**6**|['sex','bmi','bp','s1','s2','s5']|MLPRegressor(activation='logistic', alpha=0.010729877296924203, hidden_layer_sizes=1, max_iter=5000, n_iter_no_change=20, random_state=1234, solver='lbfgs', tol=1e-05)|\
|KernelRidge|0.483465|0.679036|7|['age','sex','bmi','bp','s3','s5','s6']|KernelRidge(alpha=0.3664462701238023, gamma=0.01808883688516421, kernel='rbf')|\
|SVR|0.487392|0.682699|8|['age','sex','bmi','bp','s1','s4','s5','s6']|SVR(C=0.8476135773996406, gamma=0.02324169209860404)|\
|KNeighborsRegressor|0.521326|0.687740|6|['sex','bmi','bp','s3','s5','s6']|KNeighborsRegressor(n\_neighbors=11)|\
|Lasso|0.493825|0.696194|7|['sex','bmi','bp','s1','s2','s5','s6']|Lasso(alpha=0.0002735058905983914)|\
|Ridge|0.492570|0.696273|7|['sex','bmi','bp','s1','s2','s5','s6']|Ridge(alpha=0.1321381563140431)|\
|RandomForestRegressor|0.552005|0.703769|9|['age','sex','bmi','bp','s2','s3','s4','s5','s6']|RandomForestRegressor(max_depth=17, min_samples_split=25, n_estimators=473)|\
|DecisionTreeRegressor|0.628316|0.864194|5|['age','sex','bmi','s4','s6']|DecisionTreeRegressor(max_depth=2, min_samples_split=20)|\

### Example 2: Binary Classification

This example shows how to use *HYBparsimony* in a binary classification problem with *breast_cancer* dataset. By default, method uses *LogisticRegression* algorithm and *neg_log_loss* as scoring metric.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
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
Iter = 0 -> MeanVal = -0.297448 ValBest = -0.091519  ComplexBest = 29,000,000,005.11 Time(min) = 0.006501

Running iteration 1
Best model -> Score = -0.085673 Complexity = 27,000,000,009.97 
Iter = 1 -> MeanVal = -0.117216 ValBest = -0.085673  ComplexBest = 27,000,000,009.97 Time(min) = 0.004273

...
...

Running iteration 102
Best model -> Score = -0.064557 Complexity = 11,000,000,039.47 
Iter = 102 -> MeanVal = -0.076314 ValBest = -0.066261  ComplexBest = 9,000,000,047.25 Time(min) = 0.004769

Running iteration 103
Best model -> Score = -0.064557 Complexity = 11,000,000,039.47 
Iter = 103 -> MeanVal = -0.086243 ValBest = -0.064995  ComplexBest = 11,000,000,031.2 Time(min) = 0.004591

Time limit reached. Stopped.

Best Model = LogisticRegression(C=5.92705799354935)
Selected features:['mean texture' 'mean concave points' 'radius error' 'area error'
 'compactness error' 'worst radius' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst concavity' 'worst symmetry']
Complexity = 11,000,000,039.47
5-CV logloss = 0.064557
logloss test = 0.076254
```

However, with small datasets like *breast_cancer*, it is highly recommended to use a repeated cross-validation and execute
*HYBparsimony* with different seeds in order to find the most important input features and best model hyper-parameters.

We also can compare with other algorithms using a robust cross-validation and more time.

```python
algorithms_clas = ['LogisticRegression', 'MLPClassifier', 
                'SVC', 'DecisionTreeClassifier',
                'RandomForestClassifier', 'KNeighborsClassifier']
res = []
for algo in algorithms_clas:
    print('#######################')
    print('Searching best: ', algo)
    HYBparsimony_model = HYBparsimony(algorithm=algo,
                                        features=breast_cancer.feature_names,
                                        rerank_error=0.005,
                                        cv=RepeatedKFold(n_splits=5, n_repeats=10),
                                        n_jobs=20,
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
In this example, the best model is also obtained with *LogisticRegression* but with $10$ features out of $30$.

|algo|Logloss\_10R5CV|Logloss\_Test|NFS|selected\_features|best\_model|
|-|-|-|-|-|-|
|**LogisticRegression**|0.066868|**0.079512**|**10**|['radius error','smoothness error','compactness error','worst radius','worst texture','worst perimeter','worst area','worst concavity','worst concave points','worst symmetry]'|LogisticRegression(C=2.5457613022710692)|
|SVC|0.061924|0.093283|9|['mean texture','radius error','smoothness error','compactness error','symmetry error','worst perimeter','worst concavity','worst concave points','worst fractal dimension]'|SVC(C=10.017400170851333, gamma=0.030271440833644657, probability=True)|
|MLPClassifier|0.055662|0.100951|14|['mean smoothness','mean compactness','mean concavity','texture error','area error','smoothness error','concave points error','fractal dimension error','worst radius','worst texture','worst perimeter','worst area','worst compactness','worst fractal dimension]'|MLPClassifier(activation='logistic', alpha=0.08468913411920591, hidden\_layer\_sizes=8, max\_iter=5000, n\_iter\_no\_change=20, random\_state=1234, solver='lbfgs', tol=1e-05)|
|DecisionTreeClassifier|0.214163|0.304484|7|['mean radius','mean compactness','mean concave points','worst texture','worst smoothness','worst symmetry','worst fractal dimension]'|DecisionTreeClassifier(max\_depth=2, min\_samples\_split=19)|
|RandomForestClassifier|0.098342|0.4229|12|['mean texture','mean smoothness','mean concave points','area error','smoothness error','compactness error','worst texture','worst area','worst smoothness','worst concave points','worst symmetry','worst fractal dimension]'|RandomForestClassifier(max\_depth=20, n\_estimators=126)|
|KNeighborsClassifier|0.079658|0.714111|17|['mean radius','mean texture','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','radius error','perimeter error','area error','smoothness error','compactness error','symmetry error','worst radius','worst texture','worst compactness','worst concave points]'|KNeighborsClassifier(n\_neighbors=7, p=1)|

### Example 3: Multiclass Classification

If the number of classes is greather than 2, *HYBparsimony* selects *f1\_macro* as scoring metric. In this example, we increase the number of particles to 20 with $npart=20$ and the $time\\_limit$ to 5 minutes. However, we also include an early stopping if best individual does not change in 20 iterations $early\\_stop=20$.

```python
import pandas as pd
import numpy as np
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
                                n_jobs=10, #Use 10 cores (1 core runs 1 fold)
                                npart = 20,
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
```
```
(178, 13)
3
Detected a multi-class problem. Using 'f1_macro' as default scoring function.
Running iteration 0
Best model -> Score = 0.981068 Complexity = 13,000,000,001.38 
Iter = 0 -> MeanVal = 0.759953   ValBest = 0.981068  ComplexBest = 13,000,000,001.38 Time(min) = 0.06835

Running iteration 1
Best model -> Score = 0.985503 Complexity = 11,000,000,036.33 
Iter = 1 -> MeanVal = 0.938299   ValBest = 0.985503  ComplexBest = 11,000,000,036.33 Time(min) = 0.071658

...
...

Running iteration 45
Best model -> Score = 0.99615 Complexity = 8,000,000,014.48 
Iter = 45 -> MeanVal = 0.984447   ValBest = 0.992284  ComplexBest = 8,000,000,009.54 Time(min) = 0.059787

Running iteration 46
Best model -> Score = 0.99615 Complexity = 8,000,000,014.48 
Iter = 46 -> MeanVal = 0.979013   ValBest = 0.992943  ComplexBest = 8,000,000,007.89 Time(min) = 0.056873

Early stopping reached. Stopped.


Best Model = LogisticRegression(C=1.1242464804883552)
Selected features:['alcohol' 'ash' 'alcalinity_of_ash' 'flavanoids' 'nonflavanoid_phenols'
 'color_intensity' 'hue' 'proline']
Complexity = 8,000,000,014.48
10R5-CV f1_macro = 0.99615
f1_macro test = 1.0
```

### Custom Evaluation

*HYBparsimony* uses by default sklearn's [*cross_val_score*](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) function as follows:

```python
def default_cv_score(estimator, X, y):
  return cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring)
```

By default $cv=5$, and $scoring$ is defined as *MSE* for regression problems, *log_loss* for binary classification problems, and *f1_macro* for multiclass problems. However, it is possible to choose [another scoring metric](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) defined in *scikit-learn* library or design [your own](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring). Also, the user can define a custom evaluation function.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import cross_val_score, RepeatedKFold
from hybparsimony import HYBparsimony
from sklearn.metrics import fbeta_score, make_scorer, cohen_kappa_score, log_loss, accuracy_score
import os

# load 'breast_cancer' dataset
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target 
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)

# Standarize X and y (some algorithms require that)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)


# #Example A: Using 10 folds and 'accuracy'
# ----------------------------------------
HYBparsimony_model = HYBparsimony(features=breast_cancer.feature_names,
                                scoring='accuracy',
                                cv=10,
                                n_jobs=10, #Use 10 cores (1 core run 1 fold)
                                rerank_error=0.001,
                                verbose=1)

HYBparsimony_model.fit(X_train, y_train, time_limit=0.1)
preds = HYBparsimony_model.predict(X_test)
print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
print(f'Selected features:{HYBparsimony_model.selected_features}')
print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
print(f'10R5-CV Accuracy = {round(HYBparsimony_model.best_score,6)}')
print(f'Accuracy test = {round(accuracy_score(y_test, preds),6)}')


#Example B: Using 10-repeated 5-fold CV and 'Kappa' score
# -------------------------------------------------------
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


metric_kappa = make_scorer(cohen_kappa_score, greater_is_better=True)
HYBparsimony_model = HYBparsimony(features=wine.feature_names,
                                scoring=metric_kappa,
                                cv=RepeatedKFold(n_splits=5, n_repeats=10),
                                n_jobs=10, #Use 10 cores (one core=one fold)
                                rerank_error=0.001,
                                verbose=1)
HYBparsimony_model.fit(X_train, y_train, time_limit=0.1)
print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
print(f'Selected features:{HYBparsimony_model.selected_features}')


#Example C: Using a weighted 'log_loss'
# -------------------------------------
# Assign a double weight to class one
def my_custom_loss_func(y_true, y_pred):
    sample_weight = np.ones_like(y_true)
    sample_weight[y_true==1] = 2.0
    return log_loss(y_true, y_pred, sample_weight=sample_weight)

# load 'breast_cancer' dataset
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target 
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)

# Standarize X and y (some algorithms require that)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Lower is better and 'log_loss' needs probabilities
custom_score = make_scorer(my_custom_loss_func, greater_is_better=False, needs_proba=True)
HYBparsimony_model = HYBparsimony(features=breast_cancer.feature_names,
                                scoring=custom_score,
                                rerank_error=0.001,
                                verbose=1)
HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)
print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
print(f'Selected features:{HYBparsimony_model.selected_features}')


# Example D: Using a 'custom evaluation' function
# -----------------------------------------------
def custom_fun(estimator, X, y):
    return cross_val_score(estimator, X, y, scoring="accuracy", n_jobs=10)

HYBparsimony_model = HYBparsimony(features=breast_cancer.feature_names,
                                custom_eval_fun=custom_fun,
                                rerank_error=0.001,
                                verbose=1)


HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)
preds = HYBparsimony_model.predict(X_test)
print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
print(f'Selected features:{HYBparsimony_model.selected_features}')
print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
print(f'10R5-CV Accuracy = {round(HYBparsimony_model.best_score,6)}')
print(f'Accuracy test = {round(accuracy_score(y_test, preds),6)}')
```

### Custom Search

*HYBparsimony* has predefined the most common scikit-learn algorithms as well as functions to measure [their complexity](https://github.com/jodivaso/hybparsimony/blob/master/hybparsimony/util/complexity.py) and the [hyperparameter ranges](https://github.com/jodivaso/hybparsimony/blob/master/hybparsimony/util/models.py) to search on. However, all this can be customized. 

In the following example, the dictionary *MLPRegressor_new* is defined. It consists of the following properties:
- *estimator* any machine learning algorithm compatible with scikit-learn.
- *complexity* the function that measures the complexity of the model.
- The hyperparameters of the algorithm. In this case, they can be fixed values (defined by Population.CONSTANT) such as '*solver*', '*activation*', etc.; or a search range $[min, max]$ defined by *{"range":(min, max), "type": Population.X}* and which type can be of three values: integer (Population.INTEGER), float (Population.FLOAT) or in powers of 10 (Population.POWER), i.e. $10^{[min, max]}$.

```python

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from hybparsimony import HYBparsimony, Population


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

def mlp_new_complexity(model, nFeatures, **kwargs):
    weights = [np.concatenate(model.intercepts_)]
    for wm in model.coefs_:
        weights.append(wm.flatten())
    weights = np.concatenate(weights) 
    int_comp = np.min((1E09-1,np.sum(weights**2)))
    return nFeatures*1E09 + int_comp

MLPRegressor_new = {"estimator": MLPRegressor, # The estimator
                "complexity": mlp_new_complexity, # The complexity
                "hidden_layer_sizes": {"range": (1, 5), "type": Population.INTEGER},
                "alpha": {"range": (-5, 5), "type": Population.POWER},
                "solver": {"value": "adam", "type": Population.CONSTANT},
                "learning_rate": {"value": "adaptive", "type": Population.CONSTANT},
                "early_stopping": {"value": True, "type": Population.CONSTANT},
                "validation_fraction": {"value": 0.10, "type": Population.CONSTANT},
                "activation": {"value": "tanh", "type": Population.CONSTANT},
                "n_iter_no_change": {"value": 20, "type": Population.CONSTANT},
                "tol": {"value": 1e-5, "type": Population.CONSTANT},
                "random_state": {"value": 1234, "type": Population.CONSTANT},
                "max_iter": {"value": 200, "type": Population.CONSTANT}
                }
HYBparsimony_model = HYBparsimony(algorithm=MLPRegressor_new,
                                features=diabetes.feature_names,
                                cv=RepeatedKFold(n_splits=5, n_repeats=10),
                                n_jobs= 25, #Use 25 cores (one core=one fold)
                                maxiter=2, # Extend to more generations (time consuming)
                                npart = 10,
                                rerank_error=0.001,
                                verbose=1)

# Search the best hyperparameters and features 
# (increasing 'time_limit' to improve RMSE with high consuming algorithms)
HYBparsimony_model.fit(X_train, y_train, time_limit=1.00)
preds = HYBparsimony_model.predict(X_test)
print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
print(f'Selected features:{HYBparsimony_model.selected_features}')
print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
print(f'5-CV MSE = {-round(HYBparsimony_model.best_score,6)}')
print(f'RMSE test = {round(mean_squared_error(y_test, preds, squared=False),6)}')


```

### Using AutoGluon

[This notebook](./examples/Autogluon_with_SHDD.ipynb) shows **how to reduce the input features from 85 to 44 (51.7%)** of an [AutoGluon](https://auto.gluon.ai/stable/index.html) model for the COIL2000 dataset downloaded from [openml.com](https://www.openml.org/). The difference in the 'log_loss' (with a test dataset) of the model trained with the 85 features versus the 44 features **is only 0.000312**.




References
----------

If you use this package, please cite this paper:

Divasón, J., Pernia-Espinoza, A., Martinez-de-Pison, F.J. (2023). [HYB-PARSIMONY: A hybrid approach combining Particle Swarm Optimization and Genetic Algorithms to find parsimonious models in high-dimensional datasets](https://authors.elsevier.com/sd/article/S0925-2312(23)00963-3). Neurocomputing, 560, 126840.
2023, Elsevier. [https://doi.org/10.1016/j.neucom.2023.126840](https://doi.org/10.1016/j.neucom.2023.126840)

**Bibtex**

```
@article{hybparsimony,
title = {HYB-PARSIMONY: A hybrid approach combining Particle Swarm Optimization and Genetic Algorithms
to find parsimonious models in high-dimensional datasets},
journal = {Neurocomputing},
volume = {560},
pages = {126840},
year = {2023},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2023.126840},
url = {https://www.sciencedirect.com/science/article/pii/S0925231223009633},
author = {Jose Divasón and Alpha Pernia-Espinoza and Francisco Javier Martinez-de-Pison}
}
```
