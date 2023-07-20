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
hyperparameter optimization (HO), and parsimonious model selection (PMS) based on a separate cost and complexity evaluation**. To improve the parsimony search, the hybrid method combines GA mechanisms such as selection, crossover and mutation within a PSO-based optimization algorithm that includes a strategy where the best position of each particle (thus, also the best position of each neighborhood) is computed considering not only the goodness-of-fit, but also the principle of parsimony. 

In HYBparsimony, the percentage of variables to be replaced with GA at each iteration $t$ is selected by a decreasing exponential function:
 $pcrossover=max(0.80 \cdot e^{(-\Gamma \cdot t)}, 0.10)$, that is adjusted by a $\Gamma$ parameter (by default $\Gamma$ is set to $0.50$). Thus, in the first iterations parsimony is promoted by GA mechanisms, i.e., replacing by crossover a high percentage of particles at the beginning. Subsequently, optimization with PSO becomes more relevant for the improvement of model accuracy. This differs from other hybrid methods in which the crossover is applied between the best individual position of each particle or other approaches in which the worst particles are also replaced by new particles, but at extreme positions.

Experiments show that, in general, and with a suitable $\Gamma$, HYB-PARSIMONY methodology allows to obtain better, more parsimonious and more robust models compared to other methods. It also reduces the number of iterations and, consequently, the computational effort.

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

This example shows how to search with *HYBparsimony* package for a parsimonious *KernelRidge* model (with low complexity) model, with *rbf* kernel, for the *diabetes* dataset. *HYBparsimony* searches for the best input features and *KernelRidge* hyperparameters: $alpha$ and $gamma$. Models are evaluated by default with a 5-fold CV mean squared error (*MSE*). Finally, root mean squared error (*$RMSE*) is showed with another test dataset to check the degree of generalization of the model.

In this example, *rerank\_error* is set to $0.001$, but other values could improve the balance between model complexity and accuracy. PMS considers the most parsimonious model with the fewest number of features. The default complexity is $M_c = 10^9{N_{FS}} + int_{comp}$  where ${N_{FS}}$ is the number of selected input features and $int_{comp}$ is the internal measure of model complexity, which depends on the algorithm used for training. In this example, $int_{comp}$ for *KernelRidge* is measured by the sum of the squared coefficients. Therefore, between two models with the same number of features, the smaller sum of the squared weights will determine the more parsimonious model (smaller weights reduce the propagation of perturbations).


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

Running iteration 2
Best model -> Score = -0.498697 Complexity = 7,000,001,419.98 
Iter = 2 -> MeanVal = -0.784296  ValBest = -0.498697   ComplexBest = 7,000,001,419.98 Time(min) = 0.004568

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
print(f'5-CV MAE = {-round(HYBparsimony_model.best_score,6)}')
print(f'RMSE test = {round(mean_squared_error(y_test, preds, squared=False),6)}')
```

```
Best Model = KernelRidge(alpha=0.26747155972470016, gamma=0.010478997542788611, kernel='rbf')
Selected features:['age' 'sex' 'bmi' 'bp' 's1' 's4' 's5' 's6']
Complexity = 8,000,002,199.12
5-CV MAE = 0.489457
RMSE test = 0.681918
```



### Example 2: Classification

This example shows how to search, for the *Sonar* database, a parsimony
SVM classificator with **GAparsimony** package.

In the next step, a fitness function is created using getFitness. This function return a fitness function for the `SVC` model, the `cohen_kappa_score` metric and the predefined `svm` complexity function for SVC models. We set regression to `False` beacause is classification example.

A SVM model is trained with these parameters and the selected input
features. Finally, *fitness()* returns a vector with three values:
the *kappa* statistic obtained with the mean of 10 runs of a 10-fold
cross-validation process, the *kappa* measured with the test database to
check the model generalization capability, and the model complexity. And the trained model.

The GA-PARSIMONY process begins defining the range of the SVM parameters
and their names. Also, *rerank\_error* can be tuned with different
*ga\_parsimony* runs to improve the **model generalization capability**.
In this example, *rerank\_error* has been fixed to 0.001 but other
values could improve the trade-off between model complexity and model
accuracy. For example, with *rerank\_error=0.01*, we can be interested 
in obtaining models with a smaller number of inputs with a *gamma* rounded
to two decimals.

```python
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

from GAparsimony import GAparsimony, Population, getFitness
from GAparsimony.util import svm_complexity

wine = load_wine()
X, y = wine.data, wine.target 
X = StandardScaler().fit_transform(X)


rerank_error = 0.001
params = {"C":{"range": (00.0001, 99.9999), "type": Population.FLOAT}, 
            "gamma":{"range": (0.00001,0.99999), "type": Population.FLOAT}, 
            "kernel": {"value": "poly", "type": Population.CONSTANT}}


fitness = getFitness(SVC, cohen_kappa_score, svm_complexity, minimize=False, test_size=0.2, random_state=42, n_jobs=-1)


GAparsimony_model = GAparsimony(fitness=fitness,
                                  params=params,
                                  features=wine.feature_names,
                                  keep_history = True,
                                  rerank_error = rerank_error,
                                  popSize = 40,
                                  maxiter = 50, early_stop=10,
                                  feat_thres=0.90, # Perc selected features in first generation
                                  feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                  seed_ini = 1234)
```

With small databases, it is highly recommended to execute
**GAparsimony** with different seeds in order to find
the most important input features and model parameters.

In this example, one GA optimization is presented with a training database 
composed of 60 input features and 167 instances, and a test database with only 41 instances.
Hence, a robust validation metric is necessary. Thus, a repeated cross-validation is performed.

Starts the GA optimizaton process with 40 individuals per generation and
a maximum number of 5 iterations with an early stopping when
validation measure does not increase significantly in 3 generations.
Parallel is activated. In addition, history of each iteration is saved
in order to use *plot* and *parsimony\_importance* methods.

``` {.python}
GAparsimony_model.fit(X, y)
```
```
#output

GA-PARSIMONY | iter = 0
  MeanVal = 0.8797661  |  ValBest = 0.9410622  |  TstBest = 0.9574468  |ComplexBest = 10000000045.0| Time(min) = 0.1504835  

GA-PARSIMONY | iter = 1
  MeanVal = 0.9049894  |  ValBest = 0.9456775  |     TstBest = 1.0     |ComplexBest = 11000000044.0| Time(min) = 0.0590165  

GA-PARSIMONY | iter = 2
  MeanVal = 0.9189347  |  ValBest = 0.9456775  |     TstBest = 1.0     |ComplexBest = 11000000044.0| Time(min) = 0.0520666  

GA-PARSIMONY | iter = 3
  MeanVal = 0.9270711  |   ValBest = 0.952701  |  TstBest = 0.9568345  |ComplexBest = 10000000043.0| Time(min) = 0.0494999

...

GA-PARSIMONY | iter = 28
  MeanVal = 0.9370426  |  ValBest = 0.9840488  |  TstBest = 0.9574468  |ComplexBest = 9000000052.0| Time(min) = 0.0497332  

GA-PARSIMONY | iter = 29
  MeanVal = 0.9363377  |  ValBest = 0.9840488  |  TstBest = 0.9574468  |ComplexBest = 9000000052.0| Time(min) = 0.0467499  

GA-PARSIMONY | iter = 30
  MeanVal = 0.9204895  |  ValBest = 0.9840488  |  TstBest = 0.9574468  |ComplexBest = 9000000052.0| Time(min) = 0.0500166  

GA-PARSIMONY | iter = 31
  MeanVal = 0.9466802  |  ValBest = 0.9840488  |  TstBest = 0.9574468  |ComplexBest = 9000000052.0| Time(min) = 0.0481334
```

summary() shows the GA initial settings and two solutions: the solution with the best validation score in the whole GA optimization process, and finally, the best parsimonious individual at the last generation.

``` {.python}
GAparsimony_model.summary()
```
``` 
+------------------------------------+
|             GA-PARSIMONY           |
+------------------------------------+

GA-PARSIMONY settings:
 Number of Parameters      = 2
 Number of Features        = 13
 Population size           = 40
 Maximum of generations    = 50
 Number of early-stop gen. = 10
 Elitism                   = 8
 Crossover probability     = 0.8
 Mutation probability      = 0.1
 Max diff(error) to ReRank = 0.001
 Perc. of 1s in first popu.= 0.9
 Prob. to be 1 in mutation = 0.1

 Search domain =
                 C    gamma  alcohol  malic_acid  ash  alcalinity_of_ash  \
Min_param   0.0001  0.00001      0.0         0.0  0.0                0.0
Max_param  99.9999  0.99999      1.0         1.0  1.0                1.0

           magnesium  total_phenols  flavanoids  nonflavanoid_phenols  \
Min_param        0.0            0.0         0.0                   0.0
Max_param        1.0            1.0         1.0                   1.0

           proanthocyanins  color_intensity  hue  \
Min_param              0.0              0.0  0.0
Max_param              1.0              1.0  1.0

           od280/od315_of_diluted_wines  proline
Min_param                           0.0      0.0
Max_param                           1.0      1.0


GA-PARSIMONY results:
 Iterations                = 32
 Best validation score = 0.9840488232315704


Solution with the best validation score in the whole GA process =

  fitnessVal fitnessTst complexity         C     gamma alcohol malic_acid ash  \
0   0.984049   0.957447      9e+09  0.527497  0.225906       1          1   1

  alcalinity_of_ash magnesium total_phenols flavanoids nonflavanoid_phenols  \
0                 1         0             0          1                    0

  proanthocyanins color_intensity hue od280/od315_of_diluted_wines proline
0               1               0   1                            1       1


Results of the best individual at the last generation =

 Best indiv's validat.cost = 0.9840488232315704
 Best indiv's testing cost = 0.9574468085106383
 Best indiv's complexity   = 9000000052.0
 Elapsed time in minutes   = 1.705049173037211


BEST SOLUTION =

  fitnessVal fitnessTst complexity         C     gamma alcohol malic_acid ash  \
0   0.984049   0.957447      9e+09  0.527497  0.225906       1          1   1

  alcalinity_of_ash magnesium total_phenols flavanoids nonflavanoid_phenols  \
0                 1         0             0          1                    0

  proanthocyanins color_intensity hue od280/od315_of_diluted_wines proline
0               1               0   1                            1       1
```

Plot GA evolution.

``` {.python}
GAparsimony_model.plot()
```
![GA-PARSIMONY Evolution](https://raw.githubusercontent.com/misantam/GAparsimony/main/docs/img/classification_readme.png)

GA-PARSIMONY evolution

Show percentage of appearance for each feature in elitists

``` {.python}
# Percentage of appearance for each feature in elitists
GAparsimony_model.importance()
```
```
+--------------------------------------------+
|                  GA-PARSIMONY              |
+--------------------------------------------+

Percentage of appearance of each feature in elitists:

  alcohol  ash proline flavanoids alcalinity_of_ash malic_acid  \
0     100  100     100        100           99.5968    98.7903

  od280/od315_of_diluted_wines proanthocyanins      hue nonflavanoid_phenols  \
0                      98.3871         92.7419  86.6935               28.629

  color_intensity total_phenols magnesium
0         22.1774       2.41935   2.01613
```








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
