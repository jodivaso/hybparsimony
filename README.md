# HYBparsimony

[![PyPI version](https://badge.fury.io/py/GAparsimony.svg)](https://badge.fury.io/py/GAparsimony)
[![Documentation Status](https://readthedocs.org/projects/gaparsimony/badge/?version=latest)](https://gaparsimony.readthedocs.io/en/latest/?badge=latest)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=GAparsimony&metric=alert_status)](https://sonarcloud.io/dashboard?id=GAparsimony)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=GAparsimony&metric=bugs)](https://sonarcloud.io/dashboard?id=GAparsimony)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=GAparsimony&metric=code_smells)](https://sonarcloud.io/dashboard?id=GAparsimony)


HYBparsimony
===========

[Documentation](https://gaparsimony.readthedocs.io/en/latest/index.html)

HYBparsimony for Python is a package for searching accurate parsimonious models by combining feature selection (FS), model
hyperparameter optimization (HO), and parsimonious model selection (PMS). PMS is based on separate cost and complexity evaluations. To improve the parsimony search, it is proposed a hybrid method that combines GA mechanisms such as selection, crossover and mutation within a PSO-based optimization algorithm that includes a strategy where the best position of each particle (thus, also the best position
of each neighborhood) is computed considering not only the goodness-of-fit, but also the principle of parsimony. 

In HYBparsimony the percentage of variables to be replaced at each iteration, $t$, is selected by a decreasing exponential function:
$pcrossover=max(0.80 \cdot e^{(-\Gamma \cdot t)}, 0.10)$
that is adjusted by a $\Gamma$ parameter:


Thus, in the first iterations parsimony is promoted by GA mechanisms, i.e., replacing by crossover a high percentage of particles at the beginning. Subsequently, optimization with PSO becomes more relevant for the improvement of model accuracy. This differs from other hybrid methods in which the crossover is applied between the best individual position of each particle or other approaches in which the worst particles are also replaced by new particles, but at extreme positions. Experiments show that, in general, and with a suitable $\Gamma$, the HYB-PARSIMONY methodology allows to obtain better, more parsimonious and more robust models compared to other methods. It also reduces the number of iterations and, consequently, the computational effort.

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

### Example 1: Classification

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

``` {.Python}
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

### Example 2: Regression

This example shows how to search, for the *Boston* database, a parsimonious
ANN model for regression and with **GAparsimony** package.

In the next step, a fitness function is created using getFitness. This function return a fitness function for the `Lasso` model, the `mean_squared_error`(RMSE) metric and the predefined `linearModels` complexity function for SVC models. We set regression to `True` beacause is classification example.

A Lasso model is trained with these parameters and the selected input
features. Finally, *fitness()* returns a vector with three negatives values:
the *RMSE* statistic obtained with the mean of 10 runs of a 10-fold
cross-validation process, the *RMSE* measured with the test database to
check the model generalization capability, and the model complexity. And the trained model.

The GA-PARSIMONY process begins defining the range of the SVM parameters
and their names. Also, *rerank\_error* can be tuned with different
*ga\_parsimony* runs to improve the **model generalization capability**.
In this example, *rerank\_error* has been fixed to 0.01 but other
values could improve the trade-off between model complexity and model
accuracy.

Therefore, PMS considers the most parsimonious model with the lower
number of features. Between two models with the same number of features,
the lower sum of the squared network weights will determine the most
parsimonious model (smaller weights reduce the propagation of disturbances).


``` {.python}
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_boston

from GAparsimony import GAparsimony, Population, getFitness
from GAparsimony.util import linearModels_complexity

boston = load_boston()
X, y = boston.data, boston.target 
X = StandardScaler().fit_transform(X)

# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error = 0.01

params = {"alpha":{"range": (1., 25.9), "type": Population.FLOAT}, 
            "tol":{"range": (0.0001,0.9999), "type": Population.FLOAT}}

fitness = getFitness(Lasso, mean_squared_error, linearModels_complexity, minimize=True, test_size=0.2, random_state=42, n_jobs=-1)


GAparsimony_model = GAparsimony(fitness=fitness,
                                params = params, 
                                features = boston.feature_names,
                                keep_history = True,
                                rerank_error = rerank_error,
                                popSize = 40,
                                maxiter = 5, early_stop=3,
                                feat_thres=0.90, # Perc selected features in first generation
                                feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                seed_ini = 1234)
```
``` {.python}
GAparsimony_model.fit(X, y)
```
```
#output

GA-PARSIMONY | iter = 0
 MeanVal = -79.1715225 | ValBest = -30.3297649 | TstBest = -29.2466835 |ComplexBest = 13000000021.927263| Time(min) = 0.1092269  

GA-PARSIMONY | iter = 1
 MeanVal = -55.1072918 | ValBest = -30.3251321 | TstBest = -29.2267507 |ComplexBest = 12000000022.088743| Time(min) = 0.0523999  

GA-PARSIMONY | iter = 2
 MeanVal = -34.9396425 | ValBest = -30.3166673 | TstBest = -28.8701544 |ComplexBest = 10000000021.774683| Time(min) = 0.0484501  

GA-PARSIMONY | iter = 3
 MeanVal = -38.6590874 |  ValBest = -30.144799 |  TstBest = -29.321512 |ComplexBest = 11000000022.865057| Time(min) = 0.0440666 

...

GA-PARSIMONY | iter = 21
 MeanVal = -40.5599677 | ValBest = -29.6343625 | TstBest = -29.3245345 |ComplexBest = 5000000023.114235| Time(min) = 0.0442333  

GA-PARSIMONY | iter = 22
 MeanVal = -36.0291598 | ValBest = -29.6343625 | TstBest = -29.3245345 |ComplexBest = 5000000023.114235| Time(min) = 0.0433499  

GA-PARSIMONY | iter = 23
 MeanVal = -36.6950374 | ValBest = -29.6343625 | TstBest = -29.3245345 |ComplexBest = 5000000023.114235|   Time(min) = 0.0441   

GA-PARSIMONY | iter = 24
 MeanVal = -37.4263523 | ValBest = -29.6343625 | TstBest = -29.3245345 |ComplexBest = 5000000023.114235| Time(min) = 0.0420333  
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
 Max diff(error) to ReRank = 0.01
 Perc. of 1s in first popu.= 0.9
 Prob. to be 1 in mutation = 0.1

 Search domain =
           alpha     tol  CRIM   ZN  INDUS  CHAS  NOX   RM  AGE  DIS  RAD  \
Min_param    1.0  0.0001   0.0  0.0    0.0   0.0  0.0  0.0  0.0  0.0  0.0
Max_param   25.9  0.9999   1.0  1.0    1.0   1.0  1.0  1.0  1.0  1.0  1.0

           TAX  PTRATIO    B  LSTAT
Min_param  0.0      0.0  0.0    0.0
Max_param  1.0      1.0  1.0    1.0


GA-PARSIMONY results:
 Iterations                = 25
 Best validation score = -29.634144915265725


Solution with the best validation score in the whole GA process =

  fitnessVal fitnessTst complexity    alpha       tol CRIM ZN INDUS CHAS NOX  \
0   -29.6341   -29.3465      6e+09  1.33747  0.523279    0  0     0    1   0

  RM AGE DIS RAD TAX PTRATIO  B LSTAT
0  1   1   0   0   0       1  1     1


Results of the best individual at the last generation =

 Best indiv's validat.cost = -29.634362465548378
 Best indiv's testing cost = -29.324534451958808
 Best indiv's complexity   = 5000000023.114235
 Elapsed time in minutes   = 1.167609703540802


BEST SOLUTION =

  fitnessVal fitnessTst complexity    alpha       tol CRIM ZN INDUS CHAS NOX  \
0   -29.6344   -29.3245      5e+09  1.33756  0.530282    0  0     0    0   0

  RM AGE DIS RAD TAX PTRATIO  B LSTAT
0  1   1   0   0   0       1  1     1
```

Plot GA evolution.

``` {.python}
GAparsimony_model.plot()
```
![GA-PARSIMONY Evolution](https://raw.githubusercontent.com/misantam/GAparsimony/main/docs/img/regression_readme.png)

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

  PTRATIO LSTAT   RM    B      AGE     CHAS      NOX   CRIM      ZN      DIS  \
0     100   100  100  100  93.2292  48.9583  48.9583  43.75  28.125  26.5625

       RAD    INDUS      TAX
0  13.5417  13.0208  8.33333
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
